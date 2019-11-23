import tensorrt as trt
import torch.nn.functional as F
from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from .interpolate_pb2 import interpolate_Message
import torch.nn as nn 

def get_interpolate_plugin(size, mode, align_corners):
    PLUGIN_NAME = 'interpolate'
    registry = trt.get_plugin_registry()
    creator = [c for c in registry.plugin_creator_list if c.name == PLUGIN_NAME and c.plugin_namespace == 'torch2trt'][0]
    message = interpolate_Message(size=size, mode=mode, align_corners=align_corners)
    return creator.deserialize_plugin(PLUGIN_NAME, message.SerializeToString())


@tensorrt_converter('torch.nn.functional.interpolate')
def convert_interpolate(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    try:
        mode = get_arg(ctx, 'mode', pos=3, default='nearest')
    except KeyError:
        mode = 'nearest'

    try:
        align_corners = get_arg(ctx, 'align_corners', pos=4, default=None)
    except KeyError:
        align_corners = False

    # currently only works for NCHW
    size = list(output.shape[2:])

    plugin = get_interpolate_plugin(size=size, mode=mode, align_corners=align_corners)

    layer = ctx.network.add_plugin_v2([input_trt], plugin)

    output._trt = layer.get_output(0)


class Interpolate(torch.nn.Module):
    def __init__(self, size, mode, align_corners):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, self.size, mode=self.mode, align_corners=self.align_corners)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)])
def test_interpolate_nearest():
    return Interpolate((224, 224), 'nearest', None)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)])
def test_interpolate_bilinear():
    return Interpolate((224, 224), 'bilinear', False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)])
def test_interpolate_bicubic():
    return Interpolate((224, 224), 'bicubic', False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)])
def test_interpolate_area():
    return Interpolate((56, 56), 'area', None)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)])
def test_upsample_scale_factor2():
    return nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)