from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter("torch.matmul")
@tensorrt_converter("torch.Tensor.__matmul__")
def convert_matmul(ctx):
    x = ctx.method_args[0]
    y = ctx.method_args[1]
    z = ctx.method_return

    x_trt, y_trt = add_missing_trt_tensors(ctx.network, [x, y])

    layer = ctx.network.add_matrix_multiply(
        x_trt,
        trt.MatrixOperation.NONE,
        y_trt,
        trt.MatrixOperation.NONE
    )

    z._trt = layer.get_output(0)


class Matmul(torch.nn.Module):

    def forward(self, x, y):
        return x @ y

@add_module_test(torch.float32, torch.device('cuda'), [(3, 4), (4, 3)])
def test_matmul_basic():
    return Matmul()