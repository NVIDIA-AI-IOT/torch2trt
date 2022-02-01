import torch
import torch.nn as nn
from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from torch2trt.plugins.creators.create_reflection_pad_2d_plugin import create_reflection_pad_2d_plugin


@tensorrt_converter(nn.ReflectionPad2d.forward)
def convert_reflection_pad(ctx):
    module = get_arg(ctx, 'self', pos=0, default=None)
    input = get_arg(ctx, 'x', pos=1, default=None)
    output = ctx.method_return
    input_trt = input._trt
    plugin = create_reflection_pad_2d_plugin(
        module.padding[0],
        module.padding[1],
        module.padding[2],
        module.padding[3]
    )
    layer = ctx.network.add_plugin_v2([input_trt], plugin)
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 1, 3, 3)])
@add_module_test(torch.float32, torch.device("cuda"), [(1, 2, 3, 3)])
def test_reflection_pad_2d_simple():
    return nn.ReflectionPad2d(1)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 1, 3, 3)])
@add_module_test(torch.float32, torch.device("cuda"), [(1, 2, 3, 3)])
def test_reflection_pad_2d_simple():
    return nn.ReflectionPad2d(2)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 1, 3, 3)])
@add_module_test(torch.float32, torch.device("cuda"), [(1, 2, 3, 3)])
def test_reflection_pad_2d_simple():
    return nn.ReflectionPad2d((1, 0, 1, 0))