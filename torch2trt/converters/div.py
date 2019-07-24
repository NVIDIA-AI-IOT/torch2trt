from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.Tensor.__truediv__')
def convert_div(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a._trt, input_b._trt, trt.ElementWiseOperation.DIV)
    output._trt = layer.get_output(0)


class Div(torch.nn.Module):
    def __init__(self):
        super(Div, self).__init__()

    def forward(self, x, y):
        return x / y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_div_basic():
    return Div()