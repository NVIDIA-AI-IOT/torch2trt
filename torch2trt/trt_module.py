import torch
import tensorrt as trt
from .flattener import Flattener
from .misc_utils import (
    torch_dtype_from_trt,
    torch_device_from_trt
)
from .version_utils import (
    trt_version
)


class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None, input_flattener=None, output_flattener=None):
        super(TRTModule, self).__init__()
        self._register_state_dict_hook(TRTModule._on_state_dict)

        if isinstance(engine, str):
            # assume filepath
            with open(engine, 'rb') as f:
                engine = f.read()
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(engine)
        elif isinstance(engine, trt.IHostMemory):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(engine)
            
        self.engine = engine
        if self.engine is not None:
            self.context = self.engine.create_execution_context()
            self._update_name_binindgs_maps()
        self.input_names = input_names
        self.output_names = output_names
        self.input_flattener = input_flattener
        self.output_flattener = output_flattener
    
    def _update_name_binindgs_maps(self):
        if trt_version() >= "10.0":
            self._update_name_binding_maps_trt_10()
        else:
            self._update_name_binding_maps_pre_trt_10()

    def _update_name_binding_maps_trt_10(self):
        self._name_to_binding = {}
        self._binding_to_name = {}
        for i in range(self.engine.num_io_tensors):
            name_i = self.engine.get_tensor_name(i)
            self._name_to_binding[name_i] = i
            self._binding_to_name[i] = name_i

    def _update_name_binding_maps_pre_trt_10(self):
        self._name_to_binding = {}
        self._binding_to_name = {}
        for i in range(self.engine.num_bindings):
            name_i = self.engine.get_binding_name(i)
            self._name_to_binding[name_i] = i
            self._binding_to_name[i] = name_i

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        state_dict[prefix + "engine"] = bytearray(self.engine.serialize())
        state_dict[prefix + "input_names"] = self.input_names
        state_dict[prefix + "output_names"] = self.output_names
        state_dict[prefix + "input_flattener"] = self.input_flattener.dict()
        state_dict[prefix + "output_flattener"] = self.output_flattener.dict()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        engine_bytes = state_dict[prefix + "engine"]

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            self.context = self.engine.create_execution_context()

        self.input_names = state_dict[prefix + "input_names"]
        self.output_names = state_dict[prefix + "output_names"]

        if 'input_flattener' in state_dict:
            self.input_flattener = Flattener.from_dict(state_dict['input_flattener'])
        else:
            self.input_flattener = None

        if 'output_flattener' in state_dict:
            self.output_flattener = Flattener.from_dict(state_dict['output_flattener'])
        else:
            self.output_flattener = None

        self._update_name_binindgs_maps()

    def _forward_pre_10(self, *inputs):
        bindings = [None] * (len(self.input_names) + len(self.output_names))
        
        if self.input_flattener is not None:
            inputs = self.input_flattener.flatten(inputs)

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            shape = tuple(inputs[i].shape)
            bindings[idx] = inputs[i].contiguous().data_ptr()
            self.context.set_binding_shape(idx, shape)

        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        self.context.execute_async_v2(
            bindings, torch.cuda.current_stream().cuda_stream
        )

        if self.output_flattener is not None:
            outputs = self.output_flattener.unflatten(outputs)
        else:
            outputs = tuple(outputs)
            if len(outputs) == 1:
                outputs = outputs[0]

        return outputs

    def _forward_post_10(self, *inputs):
        if self.input_flattener is not None:
            inputs = self.input_flattener.flatten(inputs)

        # set shapes
        for i, input_name in enumerate(self.input_names):
            shape = tuple(inputs[i].shape)
            data_ptr = inputs[i].contiguous().data_ptr()
            self.context.set_tensor_address(input_name, data_ptr)
            self.context.set_input_shape(input_name, shape)

        # execute
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            dtype = torch_dtype_from_trt(self.engine.get_tensor_dtype(output_name))
            shape = tuple(self.context.get_tensor_shape(output_name))
            device = torch_device_from_trt(self.engine.get_tensor_location(output_name))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            self.context.set_tensor_address(output_name, output.data_ptr())

        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

        if self.output_flattener is not None:
            outputs = self.output_flattener.unflatten(outputs)
        else:
            outputs = tuple(outputs)
            if len(outputs) == 1:
                outputs = outputs[0]

        return outputs

    def forward(self, *inputs):
        if trt_version() < "10.0":
            return self._forward_pre_10(*inputs)
        else:
            return self._forward_post_10(*inputs)

    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()
