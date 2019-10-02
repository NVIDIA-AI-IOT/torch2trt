from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.max_pool2d')
def convert_max_pool2d(ctx):
    # parse args
    input = get_arg(ctx, 'input', pos=0, default=None)
    kernel_size = get_arg(ctx, 'kernel_size', pos=1, default=None)
    stride = get_arg(ctx, 'stride', pos=2, default=None)
    padding = get_arg(ctx, 'padding', pos=3, default=0)
    dilation = get_arg(ctx, 'dilation', pos=4, default=1)
    ceil_mode = get_arg(ctx, 'ceil_mode', pos=5, default=False)
    
    # get input trt tensor (or create constant if it doesn't exist)
    input_trt = trt_(ctx.network, input)
    
    output = ctx.method_return

    # get kernel size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2

    # get stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    # get padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    layer = ctx.network.add_pooling(
        input=input_trt, type=trt.PoolingType.MAX, window_size=kernel_size)
    
    layer.stride = stride
    layer.padding = padding
    
    if ceil_mode:
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