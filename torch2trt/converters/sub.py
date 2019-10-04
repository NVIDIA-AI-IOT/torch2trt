from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.sub')
@tensorrt_converter('torch.Tensor.__isub__')
@tensorrt_converter('torch.Tensor.__sub__')
def convert_sub(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.SUB)
    output._trt = layer.get_output(0)

    
@tensorrt_converter('torch.Tensor.__rsub__')
def convert_sub(ctx):
    input_a = ctx.method_args[1]
    input_b = ctx.method_args[0]  # flipped for rsub
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.SUB)
    output._trt = layer.get_output(0)
    

class Sub(torch.nn.Module):
    def __init__(self):
        super(Sub, self).__init__()

    def forward(self, x, y):
        return x - y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_sub_basic():
    return Sub()


class ISub(torch.nn.Module):
    def __init__(self):
        super(ISub, self).__init__()

    def forward(self, x, y):
        x -= y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_sub_isub():
    return ISub()


class TorchSub(torch.nn.Module):
    def __init__(self):
        super(TorchSub, self).__init__()

    def forward(self, x, y):
        return torch.sub(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_torch_sub():
    return TorchSub()


class RSubInt(torch.nn.Module):
    def __init__(self):
        super(RSubInt, self).__init__()

    def forward(self, x):
        return 1 - x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rsub_int():
    return RSubInt()


class RSubFloat(torch.nn.Module):
    def __init__(self):
        super(RSubFloat, self).__init__()

    def forward(self, x):
        return 1.0 - x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rsub_float():
    return RSubFloat()