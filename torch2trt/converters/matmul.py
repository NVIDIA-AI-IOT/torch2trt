from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.matmul')
def convert_mul(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_matrix_multiply(input_a_trt, trt.MatrixOperation.NONE, input_b_trt, trt.MatrixOperation.NONE)
    output._trt = layer.get_output(0)


class TorchMatmul(torch.nn.Module):
    def __init__(self):
        super(TorchMatmul, self).__init__()

    def forward(self, x, y):
        result = torch.matmul(x, y)
        return result


@add_module_test(torch.float32, torch.device('cuda'), [(1,3,4), (1,4,2)])
def test_matmul1():
    return TorchMatmul()

@add_module_test(torch.float32, torch.device('cuda'), [(1,3,4), (1,5,4,2)])
def test_matmul2():
    return TorchMatmul()

