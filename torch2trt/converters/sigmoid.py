from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.sigmoid')
@tensorrt_converter('torch.sigmoid')
def convert_sigmoid(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SIGMOID)
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_sigmoid_basic():
    return torch.nn.Sigmoid()