from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

@tensorrt_converter('torch.fmod')
@tensorrt_converter('torch.Tensor.__mod__')
def convert_mod(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape) - 1)
    # a % b  =  a - (a//b) * b
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.FLOOR_DIV)
    layer = ctx.network.add_elementwise(layer.get_output(0), input_b_trt, trt.ElementWiseOperation.PROD)
    layer = ctx.network.add_elementwise(input_a_trt, layer.get_output(0), trt.ElementWiseOperation.SUB)
    output._trt = layer.get_output(0)


class Mod(torch.nn.Module):
    def __init__(self):
        super(Mod, self).__init__()

    def forward(self, x, y):
        return x % y


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20), (1, 3, 1, 20)])
def test_mod_basic_float():
    return Mod()


@add_module_test(torch.int32, torch.device('cuda'), [(1, 3, 40, 20), (1, 3, 1, 20)])
def test_mod_basic_int():
    return Mod()


class IMod(torch.nn.Module):
    def __init__(self):
        super(IMod, self).__init__()

    def forward(self, x, y):
        x %= y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20), (1, 3, 40, 20)])
def test_mod_imod():
    return IMod()


class TorchMod(torch.nn.Module):
    def __init__(self):
        super(TorchMod, self).__init__()

    def forward(self, x, y):
        return torch.fmod(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20), (1, 3, 40, 20)])
def test_mod_torch_mod():
    return TorchMod()


class ModConst(torch.nn.Module):
    def __init__(self):
        super(ModConst, self).__init__()

    def forward(self, x):
        return x % 2


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20)])
def test_mod_modconst():
    return ModConst()