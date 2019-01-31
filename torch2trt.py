import torch
import tensorrt as trt
from copy import copy
import numpy as np


# UTILITY FUNCTIONS


def torch_dtype_to_trt(dtype):
    if dtype == torch.int8:
        return trt.int8
    elif dtype == torch.int32:
        return trt.int32
    elif dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    else:
        raise TypeError('%s is not supported by tensorrt' % dtype)


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)


def torch_device_to_trt(device):
    if device.type == torch.device('cuda').type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device('cpu').type:
        return trt.TensorLocation.HOST
    else:
        return TypeError('%s is not supported by tensorrt' % device)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)



# CONVERSION REGISTRY AND HOOKS


CONVERTERS = {}


def attach_converter(ctx, method, converter):
    """Gets a function that executes PyTorch method and TensorRT converter"""

    def wrapper(*args, **kwargs):
        # run original method
        outputs = method(*args, **kwargs)

        # call conversion hook
        ctx.method_args = args
        ctx.method_kwargs = kwargs
        ctx.method_return = outputs
        #print('%s : %s' % (method.__qualname__, converter.__name__))
        converter(ctx)

        # convert to None so conversion will fail for unsupported layers
        ctx.method_args = None
        ctx.method_kwargs = None
        ctx.method_return = None

        return outputs

    return wrapper


class ConversionHook(object):
    """Attaches TensorRT converter to PyTorch method call"""

    def __init__(self, ctx, method, converter):
        self.ctx = ctx
        self.method_str = method
        self.method_impl = copy(eval(method))
        self.converter = converter

    def _set_method(self, method):
        exec('%s = method' % self.method_str)

    def __enter__(self):
        self._set_method(attach_converter(self.ctx, self.method_impl, self.converter))

    def __exit__(self, type, val, tb):
        self._set_method(self.method_impl)


class ConversionContext(object):
    def __init__(self, network, converters=CONVERTERS):
        self.network = network
        self.trt_tensors = {}
        self.method_args = None
        self.method_kwargs = None
        self.method_return = None
        self.hooks = [
            ConversionHook(self, method, converter)
            for method, converter in converters.items()
        ]

    def __enter__(self):
        for hook in self.hooks:
            hook.__enter__()
        return self

    def __exit__(self, type, val, tb):
        for hook in self.hooks:
            hook.__exit__(type, val, tb)

    def add_inputs(self, torch_inputs, names=None):
        if names is None:
            names = ['input_%d' % i for i in range(len(torch_inputs))]
        self.input_names = names

        for i, torch_input in enumerate(torch_inputs):
            if torch_input.__hash__() not in self.trt_tensors:
                trt_tensor = self.network.add_input(
                    name=names[i],
                    shape=tuple(torch_input.shape)[1:],
                    dtype=torch_dtype_to_trt(torch_input.dtype),
                )
                trt_tensor.location = torch_device_to_trt(torch_input.device)
                self.trt_tensors[torch_input.__hash__()] = trt_tensor

    def mark_outputs(self, torch_outputs, names=None):
        if names is None:
            names = ['output_%d' % i for i in range(len(torch_outputs))]
        self.output_names = names

        for i, torch_output in enumerate(torch_outputs):
            trt_tensor = self.trt_tensors[torch_output.__hash__()]
            trt_tensor.name = names[i]
            self.network.mark_output(trt_tensor)


class TRTModule(torch.nn.Module):
    def __init__(self, engine, input_names, output_names, final_shapes=None):
        self.input_names = input_names
        self.output_names = output_names
        self._trt_engine = engine
        self._trt_context = self._trt_engine.create_execution_context()
        super(TRTModule, self).__init__()
        self.final_shapes = final_shapes

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self._trt_engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self._trt_engine.get_binding_dtype(idx))
            if self.final_shapes is not None:
                shape = (batch_size, ) + self.final_shapes[i]
            else:
                shape = (batch_size, ) + tuple(self._trt_engine.get_binding_shape(idx))
            device = torch_device_from_trt(self._trt_engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        for i, input_name in enumerate(self.input_names):
            idx = self._trt_engine.get_binding_index(input_name)
            bindings[idx] = inputs[i].data_ptr()

        self._trt_context.execute(batch_size, bindings)

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs


def torch2trt(module, inputs, input_names=None, output_names=None, max_batch_size=1,
        fp16_mode=False, max_workspace_size=0):
    with trt.Logger(trt.Logger.INFO) as logger, trt.Builder(logger) as builder,\
            builder.create_network() as network, ConversionContext(network) as ctx:

        if isinstance(inputs, list):
            inputs = tuple(inputs)
        if not isinstance(inputs, tuple):
            inputs = (inputs, )
        ctx.add_inputs(inputs, input_names)

        outputs = module(*inputs)

        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        ctx.mark_outputs(outputs, output_names)

        final_shapes = [tuple(output.shape)[1:] for output in list(outputs)]

        builder.max_workspace_size = max_workspace_size
        builder.fp16_mode = fp16_mode
        builder.max_batch_size = max_batch_size

        engine = builder.build_cuda_engine(network)

    return TRTModule(engine, ctx.input_names, ctx.output_names, final_shapes)


# DEFINE ALL CONVERSION FUNCTIONS


def tensorrt_converter(method):
    def register_converter(converter):
        CONVERTERS[method] = converter
        return converter
    return register_converter


# MODULE CONVERTERS


@tensorrt_converter('torch.nn.Linear.forward')
def convert_Linear(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    output = ctx.method_return
    trt_input = ctx.trt_tensors[input.__hash__()]

    layer = ctx.network.add_fully_connected(
        input=trt_input,
        num_outputs=module.out_features,
        kernel=module.weight.detach().cpu().numpy(),
        bias=module.bias.detach().cpu().numpy())

    ctx.trt_tensors[output.__hash__()] = layer.get_output(0)


@tensorrt_converter('torch.nn.Conv2d.forward')
def convert_Conv2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    output = ctx.method_return
    trt_input = ctx.trt_tensors[input.__hash__()]

    kernel_size = module.kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2

    stride = module.stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    padding = module.padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    bias = trt.Weights()
    if module.bias is not None:
        bias = module.bias.detach().cpu().numpy()

    layer = ctx.network.add_convolution(
        input=trt_input,
        num_output_maps=module.out_channels,
        kernel_shape=kernel_size,
        kernel=module.weight.detach().cpu().numpy(),
        bias=bias)
    layer.stride = stride
    layer.padding = padding

    if module.groups is not None:
        layer.num_groups = module.groups

    ctx.trt_tensors[output.__hash__()] = layer.get_output(0)


@tensorrt_converter('torch.nn.MaxPool2d.forward')
def convert_MaxPool2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    output = ctx.method_return
    trt_input = ctx.trt_tensors[input.__hash__()]

    kernel_size = module.kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2

    stride = module.stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    padding = module.padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    layer = ctx.network.add_pooling(
        input=trt_input, type=trt.PoolingType.MAX, window_size=kernel_size)
    layer.stride = stride
    layer.padding = padding

    ctx.trt_tensors[output.__hash__()] = layer.get_output(0)


@tensorrt_converter('torch.nn.AvgPool2d.forward')
def convert_AvgPool2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    output = ctx.method_return
    trt_input = ctx.trt_tensors[input.__hash__()]

    kernel_size = module.kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2
    stride = module.stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * 2
    padding = module.padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    layer = ctx.network.add_pooling(
        input=trt_input, type=trt.PoolingType.AVERAGE, window_size=kernel_size)
    layer.stride = stride
    layer.padding = padding
    layer.average_count_excludes_padding = not module.count_include_pad

    ctx.trt_tensors[output.__hash__()] = layer.get_output(0)


@tensorrt_converter('torch.nn.AdaptiveAvgPool2d.forward')
def convert_AdaptiveAvgPool2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    output = ctx.method_return
    trt_input = ctx.trt_tensors[input.__hash__()]

    output_size = module.output_size
    if not isinstance(output_size, tuple):
        output_size = (output_size, ) * 2

    stride = (trt_input.shape[-2] // output_size[-2], trt_input.shape[-1] // output_size[-1])

    kernel_size = stride
    layer = ctx.network.add_pooling(
        input=trt_input, type=trt.PoolingType.AVERAGE, window_size=kernel_size)
    layer.stride = stride

    ctx.trt_tensors[output.__hash__()] = layer.get_output(0)


@tensorrt_converter('torch.nn.functional.adaptive_avg_pool2d')
def convert_adaptive_avg_pool2d(ctx):
    ctx.method_args = (torch.nn.AdaptiveAvgPool2d(ctx.method_args[1]), ctx.method_args[0])
    convert_AdaptiveAvgPool2d(ctx)


@tensorrt_converter('torch.nn.ReLU.forward')
def convert_ReLU(ctx):
    input = ctx.method_args[1]
    output = ctx.method_return
    trt_input = ctx.trt_tensors[input.__hash__()]
    layer = ctx.network.add_activation(
        input=trt_input, type=trt.ActivationType.RELU)
    ctx.trt_tensors[output.__hash__()] = layer.get_output(0)


@tensorrt_converter('torch.nn.functional.relu')
def convert_relu(ctx):
    ctx.method_args = (torch.nn.ReLU(),) + ctx.method_args
    convert_ReLU(ctx)


@tensorrt_converter('torch.nn.ReLU6.forward')
def convert_ReLU6(ctx):
    input = ctx.method_args[1]
    output = ctx.method_return
    trt_input = ctx.trt_tensors[input.__hash__()]

    layer = ctx.network.add_activation(
        input=trt_input, type=trt.ActivationType.RELU)
    shape = (1, ) * len(trt_input.shape)  # broadcast all dimensions
    tensor = 6.0 * torch.ones(shape, dtype=torch_dtype_from_trt(trt_input.dtype)).cpu().numpy()
    trt_6 = ctx.network.add_constant(shape, tensor)
    layer = ctx.network.add_elementwise(
        layer.get_output(0), trt_6.get_output(0), trt.ElementWiseOperation.MIN)

    ctx.trt_tensors[output.__hash__()] = layer.get_output(0)


@tensorrt_converter('torch.nn.functional.relu6')
def convert_relu6(ctx):
    ctx.method_args = (torch.nn.ReLU6(),) + ctx.method_args
    convert_ReLU6(ctx)


@tensorrt_converter('torch.nn.LogSoftmax.forward')
def convert_LogSoftmax(ctx):
    input = ctx.method_args[1]
    output = ctx.method_return
    trt_input = ctx.trt_tensors[input.__hash__()]
    layer = ctx.network.add_softmax(input=trt_input)
    layer = ctx.network.add_unary(input=layer.get_output(0),
            op=trt.UnaryOperation.LOG)
    ctx.trt_tensors[output.__hash__()] = layer.get_output(0)


@tensorrt_converter('torch.nn.Dropout.forward')
@tensorrt_converter('torch.nn.Dropout2d.forward')
@tensorrt_converter('torch.nn.Dropout3d.forward')
def convert_Identity(ctx):
    input = ctx.method_args[1]
    output = ctx.method_return
    ctx.trt_tensors[output.__hash__()] = ctx.trt_tensors[input.__hash__()]


@tensorrt_converter('torch.Tensor.view')
@tensorrt_converter('torch.nn.functional.dropout')
@tensorrt_converter('torch.nn.functional.dropout2d')
@tensorrt_converter('torch.nn.functional.dropout3d')
def convert_identity(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return
    ctx.trt_tensors[output.__hash__()] = ctx.trt_tensors[input.__hash__()]


@tensorrt_converter('torch.nn.BatchNorm2d.forward')
def convert_BatchNorm2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    output = ctx.method_return
    trt_input = ctx.trt_tensors[input.__hash__()]

    scale = module.weight.detach().cpu().numpy() / np.sqrt(module.running_var.detach().cpu().numpy() + module.eps)
    bias = module.bias.detach().cpu().numpy() - module.running_mean.detach().cpu().numpy() * scale
    trt_scale = ctx.network.add_constant((scale.size, 1, 1), scale)
    trt_bias = ctx.network.add_constant((bias.size, 1, 1), bias)
    layer = ctx.network.add_elementwise(trt_input, trt_scale.get_output(0), trt.ElementWiseOperation.PROD)
    layer = ctx.network.add_elementwise(layer.get_output(0), trt_bias.get_output(0), trt.ElementWiseOperation.SUM)

    ctx.trt_tensors[output.__hash__()] = layer.get_output(0)


# TENSOR METHOD CONVERTERS


@tensorrt_converter('torch.cat')
def convert_cat(ctx):
    inputs = ctx.method_args[0]

    if 'dim' in ctx.method_kwargs:
        dim = ctx.method_kwargs['dim']
    else:
        dim = ctx.method_args[1]

    output = ctx.method_return
    trt_inputs = [ctx.trt_tensors[i.__hash__()] for i in inputs]

    layer = ctx.network.add_concatenation(inputs=trt_inputs)
    layer.axis = dim - 1
    ctx.trt_tensors[output.__hash__()] = layer.get_output(0)


@tensorrt_converter('torch.Tensor.__iadd__')
@tensorrt_converter('torch.Tensor.__add__')
def convert_add(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    trt_input_a = ctx.trt_tensors[input_a.__hash__()]
    trt_input_b = ctx.trt_tensors[input_b.__hash__()]
    layer = ctx.network.add_elementwise(trt_input_a, trt_input_b, trt.ElementWiseOperation.SUM)
    ctx.trt_tensors[output.__hash__()] = layer.get_output(0)
