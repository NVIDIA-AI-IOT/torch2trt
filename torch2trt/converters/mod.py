from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.fmod')
def convert_mod(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape) - 1)
    # we can not use ElementWiseOperation.FLOOR_DIV directly because Torch truncate negative result toward 0
    # but TensorRT FLOOR_DIV op toward -Inf
    # sign = ab / |ab|
    # floordiv result: sign * (|a| // |b|)
    ab_layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.PROD)
    abs_ab_layer = ctx.network.add_unary(ab_layer.get_output(0), trt.UnaryOperation.ABS)
    sign_layer = ctx.network.add_elementwise(ab_layer.get_output(0), abs_ab_layer.get_output(0),
                                             trt.ElementWiseOperation.DIV)
    abs_a_layer = ctx.network.add_unary(input_a_trt, trt.UnaryOperation.ABS)
    abs_b_layer = ctx.network.add_unary(input_b_trt, trt.UnaryOperation.ABS)
    abs_floor_layer = ctx.network.add_elementwise(abs_a_layer.get_output(0), abs_b_layer.get_output(0),
                                                  trt.ElementWiseOperation.FLOOR_DIV)
    # a % b  =  a - (a//b) * b
    floordiv_layer = ctx.network.add_elementwise(sign_layer.get_output(0), abs_floor_layer.get_output(0),
                                            trt.ElementWiseOperation.PROD)
    prod_layer = ctx.network.add_elementwise(floordiv_layer.get_output(0), input_b_trt, trt.ElementWiseOperation.PROD)
    sub_layer = ctx.network.add_elementwise(input_a_trt, prod_layer.get_output(0), trt.ElementWiseOperation.SUB)
    output._trt = sub_layer.get_output(0)


@tensorrt_converter('torch.Tensor.__mod__')
# we need separate converter for operator because for some reason Torch use truncation toward -Inf for this op.
# bug is filed: https://github.com/pytorch/pytorch/issues/52425
# but for now we have to convert model exactly
def convert_mod(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape) - 1)
    # a % b  =  a - (a//b) * b
    floordiv_layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.FLOOR_DIV)
    prod_layer = ctx.network.add_elementwise(floordiv_layer.get_output(0), input_b_trt, trt.ElementWiseOperation.PROD)
    mod_layer = ctx.network.add_elementwise(input_a_trt, prod_layer.get_output(0), trt.ElementWiseOperation.SUB)
    output._trt = mod_layer.get_output(0)


class Mod(torch.nn.Module):
    def __init__(self):
        super(Mod, self).__init__()

    def forward(self, x, y):
        return x % y


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20), (1, 3, 1, 20)])
def test_mod_op():
    return Mod()


class ModAssign(torch.nn.Module):
    def __init__(self):
        super(ModAssign, self).__init__()

    def forward(self, x, y):
        x %= y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20), (1, 3, 1, 20)])
def test_mod_op_assign():
    return ModAssign()


class ModConst(torch.nn.Module):
    def __init__(self):
        super(ModConst, self).__init__()

    def forward(self, x):
        return x % 2.


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20)])
def test_mod_op_const():
    return ModConst()


class TorchMod(torch.nn.Module):
    def __init__(self):
        super(TorchMod, self).__init__()

    def forward(self, x, y):
        return torch.fmod(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20), (1, 3, 40, 20)])
def test_mod_func():
    return TorchMod()
