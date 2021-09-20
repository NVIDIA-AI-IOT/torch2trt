from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.zeros')
def convert_zeros(ctx):
    size = ctx.method_args
    output = ctx.method_return

    kwargs = ctx.method_kwargs
    out = kwargs.get('out') # Ignored.
    dtype = kwargs.get('dtype')
    layout = kwargs.get('layout', torch.strided) # Ignored.
    device = kwargs.get('device') # Ignored.
    requires_grad = kwargs.get('requires_grad', False) # Ignored.

    zeros_tensor = torch.zeros(*size, dtype=dtype)
    output._trt = add_trt_constant(ctx.network, zeros_tensor)


class Zeros(torch.nn.Module):
    def __init__(self, *size, dtype=None):
        super().__init__()
        self.size = size
        self.dtype = dtype

    def forward(self, x):
        return x + torch.zeros(*self.size, dtype=self.dtype, device=torch.device('cuda'))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 3, 4)])
def test_zeros_basic():
    return Zeros((1, 2, 3, 4))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 3, 4)])
def test_zeros_var_args():
    return Zeros(1, 2, 3, 4)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 2, 3, 4)])
def test_zeros_float16():
    return Zeros(1, 2, 3, 4, dtype=torch.float16)


# The fails with the following error:
# [TensorRT] ERROR: [CONSTANT #1] torch.zeros(1, 2, 3, 4, dtype=torch.int8, device=cuda, requires_grad=False): invalid weights type of Int8
#
#  @add_module_test(torch.int8, torch.device('cuda'), [(1, 2, 3, 4)])
#  def test_zeros_int8():
    #  return Zeros(1, 2, 3, 4, dtype=torch.int8)
