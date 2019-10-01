from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.Tensor.permute')
def convert_transpose(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return
    # permutation -1 because TRT does not include batch dim
    permutation = list(range(len(input.shape) - 1))
    dim0 = ctx.method_args[1] - 1
    dim1 = ctx.method_args[2] - 1
    dim2 = ctx.method_args[3] - 1
    dim3 = ctx.method_args[4] - 1
    if dim0 != -1:
        raise ValueError("Batch dimension change is not supported")

    permutation = trt.Permutation([dim1, dim2, dim3])
    layer = ctx.network.add_shuffle(input._trt)
    layer.first_transpose = permutation
    output._trt = layer.get_output(0)


class Permute(torch.nn.Module):
    def __init__(self, dim0, dim1, dim2, dim3):
        super(Permute, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
    def forward(self, x):
        return x.permute(self.dim0, self.dim1, self.dim2, self.dim3).contiguous()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_permute_0312():
    return Permute(0, 3, 1, 2)
