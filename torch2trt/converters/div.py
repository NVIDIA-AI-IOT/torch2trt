from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.div')
@tensorrt_converter('torch.Tensor.__div__') # py2
@tensorrt_converter('torch.Tensor.__idiv__') # py2
@tensorrt_converter('torch.Tensor.__truediv__') # py3
@tensorrt_converter('torch.Tensor.__itruediv__') # py3
def convert_div(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.DIV)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.Tensor.__rdiv__') # py2
@tensorrt_converter('torch.Tensor.__rtruediv__') # py3
def convert_rdiv(ctx):
    input_a = ctx.method_args[1]  # inputs switched for rdiv
    input_b = ctx.method_args[0]
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.DIV)
    output._trt = layer.get_output(0)
    

class Div(torch.nn.Module):
    def __init__(self):
        super(Div, self).__init__()

    def forward(self, x, y):
        return x / y


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_div_basic():
    return Div()


class IDiv(torch.nn.Module):
    def __init__(self):
        super(IDiv, self).__init__()

    def forward(self, x, y):
        x /= y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_div_idiv():
    return IDiv()


class TorchDiv(torch.nn.Module):
    def __init__(self):
        super(TorchDiv, self).__init__()

    def forward(self, x, y):
        return torch.div(x, y)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_div_torchdiv():
    return TorchDiv()


class RDivInt(torch.nn.Module):
    def __init__(self):
        super(RDivInt, self).__init__()

    def forward(self, x):
        return 100 / x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_rdiv_int():
    return RDivInt()


class RDivFloat(torch.nn.Module):
    def __init__(self):
        super(RDivFloat, self).__init__()

    def forward(self, x):
        return 100.0 / x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_rdiv_float():
    return RDivFloat()