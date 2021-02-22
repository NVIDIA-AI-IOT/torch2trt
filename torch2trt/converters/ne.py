from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.ne')
@tensorrt_converter('torch.Tensor.__ne__')
def convert_ne(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape) - 1)
    layer_1 = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.EQUAL)
    layer_2 = ctx.network.add_unary(layer_1.get_output(0), trt.UnaryOperation.NOT)
    output._trt = layer_2.get_output(0)


class NotEqual(torch.nn.Module):
    def __init__(self):
        super(NotEqual, self).__init__()

    def forward(self, x, y):
        return x != y


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20), (1, 3, 1, 20)])
def test_ne_op():
    return NotEqual()


class NotEqualConst(torch.nn.Module):
    def __init__(self):
        super(NotEqualConst, self).__init__()

    def forward(self, x):
        return x != 13.62


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20)])
def test_ne_op_const():
    return NotEqualConst()


class TorchNotEqual(torch.nn.Module):
    def __init__(self):
        super(TorchNotEqual, self).__init__()

    def forward(self, x, y):
        return torch.ne(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20), (1, 3, 1, 20)])
def test_ne_torch():
    return TorchNotEqual()
