from torch2trt.torch2trt import *


@tensorrt_converter('torch.nn.AdaptiveAvgPool2d.forward')
def convert_AdaptiveAvgPool2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    output = ctx.method_return

    output_size = module.output_size
    if not isinstance(output_size, tuple):
        output_size = (output_size, ) * 2

    stride = (input._trt.shape[-2] // output_size[-2], input._trt.shape[-1] // output_size[-1])

    kernel_size = stride
    layer = ctx.network.add_pooling(
        input=input._trt, type=trt.PoolingType.AVERAGE, window_size=kernel_size)
    layer.stride = stride

    output._trt = layer.get_output(0)