import torch
import torch.nn as nn
from torch2trt.torch2trt import *
import numpy as np
import ctypes


try:
    ctypes.CDLL('libtorch2trt_plugins.so')

    def create_reflection_pad_2d_plugin(paddingLeft, paddingRight, paddingTop, paddingBottom):

        registry = trt.get_plugin_registry()
        creator = registry.get_plugin_creator('ReflectionPad2dPlugin', '1', '')

        fc = trt.PluginFieldCollection([
            trt.PluginField(
                'paddingLeft',
                np.array([paddingLeft]).astype(np.int32),
                trt.PluginFieldType.INT32
            ),
            trt.PluginField(
                'paddingRight',
                np.array([paddingRight]).astype(np.int32),
                trt.PluginFieldType.INT32
            ),
            trt.PluginField(
                'paddingTop',
                np.array([paddingTop]).astype(np.int32),
                trt.PluginFieldType.INT32
            ),
            trt.PluginField(
                'paddingBottom',
                np.array([paddingBottom]).astype(np.int32),
                trt.PluginFieldType.INT32
            )
        ])

        return creator.create_plugin('', fc)
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

except:
    pass