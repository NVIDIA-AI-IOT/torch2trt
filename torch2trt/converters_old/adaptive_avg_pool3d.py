from torch2trt.torch2trt import *
from .AdaptiveAvgPool3d import *


@tensorrt_converter("torch.nn.functional.adaptive_avg_pool3d")
def convert_adaptive_avg_pool3d(ctx):
    ctx.method_args = (
        torch.nn.AdaptiveAvgPool3d(ctx.method_args[1]),
        ctx.method_args[0],
    )
    convert_AdaptiveAvgPool3d(ctx)
