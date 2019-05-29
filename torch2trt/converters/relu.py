from torch2trt.torch2trt import *
from .ReLU import *


@tensorrt_converter('torch.nn.functional.relu')
def convert_relu(ctx):
    ctx.method_args = (torch.nn.ReLU(),) + ctx.method_args
    convert_ReLU(ctx)