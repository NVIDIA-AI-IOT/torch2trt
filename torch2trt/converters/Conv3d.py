import tensorrt as trt
import torch.nn.functional as F
from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from torch2trt.plugins.utils import create_plugin
from torch2trt.plugins.convnd_plugin_pb2 import ConvNdPluginMsg


@tensorrt_converter('torch.nn.Conv3d.forward')
def convert_Conv3d(ctx):
#     import pdb
#     pdb.set_trace()
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    output = ctx.method_return

    DIM = 3
    
    kernel_size = module.kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * DIM

    stride = module.stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * DIM

    padding = module.padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * DIM

    dilation = module.dilation
    if not isinstance(dilation, tuple):
        dilation = (dilation, ) * DIM
        
    groups = module.groups

    weight = module.weight
    weight._trt = ctx.network.add_constant(tuple(weight.shape), weight.detach().cpu().numpy()).get_output(0)
    
    has_bias = False
    if module.bias is not None:
        has_bias = True
        bias = module.bias
        bias._trt = ctx.network.add_constant(tuple(bias.shape), bias.detach().cpu().numpy()).get_output(0)

    method = ConvNdPluginMsg(
        num_dim = DIM,
        kernel_size = kernel_size,
        stride = stride,
        padding = padding,
        dilation = dilation,
        groups = groups,
        transposed = False,
        has_bias = has_bias
    )
    
    inputs = [input, weight]
    if module.bias is not None:
        inputs += [bias]
    
    trt_inputs = [inp._trt for inp in inputs]
        
    plugin = create_plugin('convnd', method, inputs, [output])

    layer = ctx.network.add_plugin_v2(trt_inputs, plugin)

    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 30, 30, 30)])
def test_conv3d_basic():
    return torch.nn.Conv3d(5, 5, 3)