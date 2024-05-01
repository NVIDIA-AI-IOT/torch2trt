from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.Tensor.__floordiv__')
@tensorrt_converter('torch.Tensor.__ifloordiv__')
@tensorrt_converter('torch.floor_divide')
def convert_floordiv(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))
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
    out_layer = ctx.network.add_elementwise(sign_layer.get_output(0), abs_floor_layer.get_output(0),
                                            trt.ElementWiseOperation.PROD)
    output._trt = out_layer.get_output(0)


class FloorDiv(torch.nn.Module):
    def __init__(self):
        super(FloorDiv, self).__init__()

    def forward(self, x, y):
        return x // y


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20), (1, 3, 1, 20)])
def test_floordiv_op():
    return FloorDiv()


class FloorDivAssign (torch.nn.Module):
    def __init__(self):
        super(FloorDivAssign, self).__init__()

    def forward(self, x, y):
        x //= y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20), (1, 3, 1, 20)])
def test_floordiv_op_assign():
    return FloorDivAssign()


class FloorDivConst(torch.nn.Module):
    def __init__(self):
        super(FloorDivConst, self).__init__()

    def forward(self, x):
        return x // 2.


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20)])
def test_floordiv_op_const():
    return FloorDivConst()


class TorchFloorDiv(torch.nn.Module):
    def __init__(self):
        super(TorchFloorDiv, self).__init__()

    def forward(self, x, y):
        return torch.floor_divide(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20), (1, 3, 1, 20)])
def test_floordiv_func():
    return TorchFloorDiv()
