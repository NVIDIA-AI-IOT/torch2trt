from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


def _set_layer_precision(ctx, layer):
    # Supported TRT precisions as given by torch2trt_kwargs.
    INT8_MODE = "int8_mode"
    FP16_MODE = "fp16_mode"

    # Check that args exist as expected in torch2trt_kwargs.
    trt_kwargs = ctx.torch2trt_kwargs
    assert INT8_MODE in trt_kwargs
    assert FP16_MODE in trt_kwargs

    is_int8 = trt_kwargs.get(INT8_MODE, False)
    is_fp16 = trt_kwargs.get(FP16_MODE, False)

    if is_int8:
        layer.precision = trt.int8
        layer.set_output_type(0, trt.int8)
    elif is_fp16:
        layer.precision = trt.float16
        layer.set_output_type(0, trt.float16)


@tensorrt_converter('torch.tril_indices')
def convert_tril_indices(ctx):
    tensor = ctx.method_return

    # Implementation mostly copied from add_trt_constant.
    array = tensor.detach().cpu().numpy()
    layer = ctx.network.add_constant(tensor.shape, array)

    _set_layer_precision(ctx, layer)

    tensor._trt = layer.get_output(0)


class TrilIndices(torch.nn.Module):
    def __init__(self, row, col, offset=0):
        super().__init__()
        self.row = row
        self.col = col
        self.offset = offset

    def forward(self, x):
        return x + torch.tril_indices(self.row, self.col, offset=self.offset, dtype=torch.float32, device=torch.device('cuda'))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 6)])
def test_tril_indices_basic():
    return TrilIndices(3, 3)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 6)], fp16_mode=True)
def test_tril_indices_fp16_mode():
    return TrilIndices(3, 3)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 6)], int8_mode=True)
def test_tril_indices_int8_mode():
    return TrilIndices(3, 3)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 6)])
def test_tril_indices_negative_offset():
    return TrilIndices(4, 3, -1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 11)])
def test_tril_indices_positive_offset():
    return TrilIndices(4, 3, 1)

