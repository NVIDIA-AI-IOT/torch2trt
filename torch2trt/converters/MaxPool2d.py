from torch2trt.torch2trt import *


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