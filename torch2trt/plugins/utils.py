import tensorrt as trt
from .torch_plugin_pb2 import TorchPluginMsg


def create_plugin(plugin_name, method_msg, inputs, outputs):
    msg = TorchPluginMsg()
    
    # add shapes
    for input in inputs:
        msg.input_shapes.add(size=tuple(input.shape[1:])) # exclude batch dimension
    for output in outputs:
        msg.output_shapes.add(size=tuple(output.shape[1:])) # exclude batch dimension
        
    # pack method
    msg.method.Pack(method_msg)
    
    # find in registry
    registry = trt.get_plugin_registry()
    creator = [c for c in registry.plugin_creator_list if c.name == plugin_name and c.plugin_namespace == 'torch2trt'][0]
    
    return creator.deserialize_plugin(plugin_name, msg.SerializeToString())