from torch2trt.torch2trt import *
from .AdaptiveMaxPool2d import *


@tensorrt_converter('torch.nn.functional.adaptive_max_pool2d')
def convert_adaptive_avg_pool2d(ctx):
    ctx.method_args = (torch.nn.AdaptiveMaxPool2d(ctx.method_args[1]), ctx.method_args[0])
    convert_AdaptiveMaxPool2d(ctx)
