from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.ReLU6.forward')
def convert_ReLU6(ctx):
    input = ctx.method_args[1]
    output = ctx.method_return
    
    input_trt, trt_6 = trt_(ctx.network, input, 6)

    layer = ctx.network.add_activation(
        input=input_trt, type=trt.ActivationType.RELU)
    layer = ctx.network.add_elementwise(
        layer.get_output(0), trt_6, trt.ElementWiseOperation.MIN)

    output._trt = layer.get_output(0)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_relu6_basic():
    return torch.nn.ReLU6()