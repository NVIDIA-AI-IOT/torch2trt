from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.mul')
@tensorrt_converter('torch.Tensor.__imul__')
@tensorrt_converter('torch.Tensor.__mul__')
@tensorrt_converter('torch.Tensor.__rmul__')
def convert_mul(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.PROD)
    output._trt = layer.get_output(0)

class Mul(torch.nn.Module):
    def __init__(self):
        super(Mul, self).__init__()

    def forward(self, x, y):
        return x * y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_mul_basic():
    return Mul()


class IMul(torch.nn.Module):
    def __init__(self):
        super(IMul, self).__init__()

    def forward(self, x, y):
        x *= y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_mul_imul():
    return IMul()


class TorchMul(torch.nn.Module):
    def __init__(self):
        super(TorchMul, self).__init__()

    def forward(self, x, y):
        return torch.mul(x, y)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_mul_torchmul():
    return TorchMul()


class RMulInt(torch.nn.Module):
    def __init__(self):
        super(RMulInt, self).__init__()

    def forward(self, x):
        return 10 * x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_rmul_int():
    return RMulInt()


class RMulFloat(torch.nn.Module):
    def __init__(self):
        super(RMulFloat, self).__init__()

    def forward(self, x):
        return 10.0 * x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_rmul_float():
    return RMulFloat()