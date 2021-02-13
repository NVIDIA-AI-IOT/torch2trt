from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

@tensorrt_converter('torch.floor_divide')
@tensorrt_converter('torch.Tensor.__floordiv__')
def convert_floordiv(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape) - 1)
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.FLOOR_DIV)
    output._trt = layer.get_output(0)


class FloorDiv(torch.nn.Module):
    def __init__(self):
        super(FloorDiv, self).__init__()

    def forward(self, x, y):
        return x // y


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20), (1, 3, 1, 20)])
def test_floordiv_basic():
    return FloorDiv()


class TorchFloorDiv(torch.nn.Module):
    def __init__(self):
        super(TorchFloorDiv, self).__init__()

    def forward(self, x, y):
        return torch.floor_divide(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20), (1, 3, 1, 20)])
def test_floordiv_torch():
    return TorchFloorDiv()