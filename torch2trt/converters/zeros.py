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


@tensorrt_converter('torch.zeros')
def convert_zeros(ctx):
    tensor = ctx.method_return

    # Implementation copied from add_trt_constant.
    shape = tuple(tensor.shape[1:])
    array = tensor[0].detach().cpu().numpy()
    layer = ctx.network.add_constant(shape, array)

    _set_layer_precision(ctx, layer)

    tensor._trt = layer.get_output(0)


class Zeros(torch.nn.Module):
    def __init__(self, *size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x + torch.zeros(*self.size, device=torch.device('cuda'))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 3, 4)])
def test_zeros():
    return Zeros((1, 2, 3, 4))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 3, 4)])
def test_zeros_var_args():
    return Zeros(1, 2, 3, 4)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 3, 4)], fp16_mode=True)
def test_zeros_fp16_mode():
    return Zeros(1, 2, 3, 4)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 3, 4)], int8_mode=True)
def test_zeros_int8_mode():
    return Zeros(1, 2, 3, 4)
