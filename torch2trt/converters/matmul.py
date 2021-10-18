from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.matmul')
@tensorrt_converter('torch.mm')
@tensorrt_converter('torch.bmm')
def convert_matmul(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_matrix_multiply(input_a_trt, trt.MatrixOperation.NONE, input_b_trt, trt.MatrixOperation.NONE)
    output._trt = layer.get_output(0)


class MatMul(torch.nn.Module):
    def __init__(self):
        super(MatMul, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4), (1, 4, 5)])
def test_matmul_basic():
    return MatMul()


class BatchMatMul(torch.nn.Module):
    def __init__(self):
        super(BatchMatMul, self).__init__()

    def forward(self, x, y):
        return torch.bmm(x, y)

@add_module_test(torch.float32, torch.device('cuda'), [(10, 3, 4), (10, 4, 5)], max_batch_size=10)
def test_batchmatmul_basic():
    return BatchMatMul()