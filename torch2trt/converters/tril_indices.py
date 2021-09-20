from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.tril_indices')
def convert_tril_indices(ctx):
    row = get_arg(ctx, 'row', 0, 0)
    col = get_arg(ctx, 'col', 1, 0)
    output = ctx.method_return

    offset = get_arg(ctx, 'offset', 2, 0)

    # torch.long is pytorch's default dtype for torch.tril_indices, but is not supported by tensorrt.
    # We maintain the same default type here to allow tensorrt to error so that the user can explicitly
    # set a different dtype instead of us implicitly changing dtypes without the user's knowledge.
    dtype = get_arg(ctx, 'dtype', 3, torch.long)

    device = get_arg(ctx, 'device', 4, None) # Ignored.
    layout = get_arg(ctx, 'layout', 5, torch.strided) # Ignored.

    tril_indices = torch.tril_indices(row, col, offset=offset, dtype=dtype)
    output._trt = add_trt_constant(ctx.network, tril_indices)


class TrilIndices(torch.nn.Module):
    def __init__(self, row, col, offset=0, dtype=torch.long):
        super().__init__()
        self.row = row
        self.col = col
        self.offset = offset
        self.dtype = dtype

    def forward(self, x):
        return x + torch.tril_indices(self.row, self.col, offset=self.offset, dtype=self.dtype, device=torch.device('cuda'))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 6)])
def test_tril_indices_basic():
    return TrilIndices(3, 3, dtype=torch.float32)


# This fails with the following error:
# TypeError: torch.int64 is not supported by tensorrt
#
#  @add_module_test(torch.long, torch.device('cuda'), [(1, 2, 6)])
#  def test_tril_indices_basic():
    #  return TrilIndices(3, 3)


# This fails with the following error:
# RuntimeError: "tril_indices" not implemented for 'Half'
#
#  @add_module_test(torch.float16, torch.device('cuda'), [(1, 2, 6)])
#  def test_tril_indices_float16():
    #  return TrilIndices(3, 3, dtype=torch.float16)


# This fails with the following error:
# [TensorRT] ERROR: [CONSTANT #1] torch.tril_indices(3, 3, offset=0, dtype=torch.int8, device=cuda): invalid weights type of Int8
#
#  @add_module_test(torch.int8, torch.device('cuda'), [(1, 2, 6)])
#  def test_tril_indices_int8():
    #  return TrilIndices(3, 3, dtype=torch.int8)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 6)])
def test_tril_indices_negative_offset():
    return TrilIndices(4, 3, -1, dtype=torch.float32)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 11)])
def test_tril_indices_positive_offset():
    return TrilIndices(4, 3, 1, dtype=torch.float32)

