from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.MaxPool2d.forward')
def convert_MaxPool2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    output = ctx.method_return

    kernel_size = module.kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2

    stride = module.stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    padding = module.padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    layer = ctx.network.add_pooling(
        input=input._trt, type=trt.PoolingType.MAX, window_size=kernel_size)
    layer.stride = stride
    layer.padding = padding
    if module.ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 6)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 5, 7)])
def test_MaxPool2d_without_ceil_mode():
    return torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 6)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 5, 7)])
def test_MaxPool2d_with_ceil_mode():
    return torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
