from torch2trt.torch2trt import *
from .UPSAMPLE import * 

@tensorrt_converter('torch.nn.functional.interpolate')
def convert_interpolate(ctx):
    ctx.method_args = (torch.nn.Upsample(ctx.method_args[1]), ctx.method_args[0])
    convert_upsample(ctx)
    