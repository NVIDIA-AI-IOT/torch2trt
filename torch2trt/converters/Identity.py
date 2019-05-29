from torch2trt.torch2trt import *


@tensorrt_converter('torch.nn.Dropout.forward')
@tensorrt_converter('torch.nn.Dropout2d.forward')
@tensorrt_converter('torch.nn.Dropout3d.forward')
def convert_Identity(ctx):
    input = ctx.method_args[1]
    output = ctx.method_return
    output._trt = input._trt