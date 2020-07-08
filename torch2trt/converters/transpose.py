from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter("torch.transpose", enabled=trt_version() < '7.0')
def convert_transpose(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    # permutation -1 because TRT does not include batch dim
    permutation = list(range(len(input.shape) - 1))
    dim0 = ctx.method_args[1] - 1
    dim1 = ctx.method_args[2] - 1
    permutation[dim0] = dim1
    permutation[dim1] = dim0
    layer = ctx.network.add_shuffle(input_trt)
    layer.second_transpose = tuple(permutation)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.transpose', enabled=trt_version() >= '7.0')
def convert_transpose_trt7(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    # permutation -1 because TRT does not include batch dim
    permutation = list(range(len(input.shape) - 1))
    dim0 = ctx.method_args[1] - 1
    dim1 = ctx.method_args[2] - 1
    permutation[dim0] = dim1
    permutation[dim1] = dim0
    layer = ctx.network.add_shuffle(input_trt)
    layer.second_transpose = tuple(permutation)
    output._trt = layer.get_output(0)



class Transpose(torch.nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1).contiguous()


@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 3, 3)])
def test_transpose_12():
    return Transpose(1, 2)
