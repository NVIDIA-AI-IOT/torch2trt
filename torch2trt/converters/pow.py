from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.pow')
@tensorrt_converter('torch.Tensor.__ipow__')
@tensorrt_converter('torch.Tensor.__pow__')
def convert_pow(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.POW)
    output._trt = layer.get_output(0)

    
@tensorrt_converter('torch.Tensor.__rpow__')
def convert_pow(ctx):
    input_a = ctx.method_args[1]
    input_b = ctx.method_args[0]  # flipped for rpow
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.POW)
    output._trt = layer.get_output(0)
    

class Pow(torch.nn.Module):
    def __init__(self):
        super(Pow, self).__init__()

    def forward(self, x, y):
        return x ** y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_pow_basic():
    return Pow()


# __ipow__ not yet impl in torch
# class IPow(torch.nn.Module):
#     def __init__(self):
#         super(IPow, self).__init__()

#     def forward(self, x, y):
#         x **= y
#         return x


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
# def test_pow_ipow():
#     return IPow()


class TorchPow(torch.nn.Module):
    def __init__(self):
        super(TorchPow, self).__init__()

    def forward(self, x, y):
        return torch.pow(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_torch_pow():
    return TorchPow()


class RpowInt(torch.nn.Module):
    def __init__(self):
        super(RpowInt, self).__init__()

    def forward(self, x):
        return 2 ** x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rpow_int():
    return RpowInt()


class RpowFloat(torch.nn.Module):
    def __init__(self):
        super(RpowFloat, self).__init__()

    def forward(self, x):
        return 2.0 ** x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rpow_float():
    return RpowFloat()