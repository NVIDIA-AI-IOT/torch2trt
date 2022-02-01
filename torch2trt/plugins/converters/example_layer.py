import torch
import torch.nn as nn
import numpy as np
import tensorrt as trt
import ctypes
from torch2trt import torch2trt, tensorrt_converter, get_arg
from torch2trt_plugins.creators.example import create_ExamplePlugin


load = ctypes.CDLL('./build/libtorch2trt_plugins.so')


class ExampleLayer(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        
    def forward(self, x):
        return self.scale * x


@tensorrt_converter(ExampleLayer.forward)
def convert_example_layer(ctx):
    module = get_arg(ctx, 'self', pos=0, default=None)
    input = get_arg(ctx, 'x', pos=1, default=None)
    output = ctx.method_return
    input_trt = input._trt
    layer = ctx.network.add_plugin_v2([input_trt], create_ExamplePlugin(scale=module.scale))
    output._trt = layer.get_output(0)


