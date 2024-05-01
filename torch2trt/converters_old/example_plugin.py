import torch
import torch.nn as nn
from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
import numpy as np
import ctypes


try:
    ctypes.CDLL('libtorch2trt_plugins.so')

    def create_example_plugin(scale):

        registry = trt.get_plugin_registry()
        creator = registry.get_plugin_creator('ExamplePlugin', '1', '')

        fc = trt.PluginFieldCollection([
            trt.PluginField(
                'scale',
                scale * np.ones((1,)).astype(np.float32),
                trt.PluginFieldType.FLOAT32
            )
        ])

        return creator.create_plugin('', fc)

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
        plugin = create_example_plugin(module.scale)
        layer = ctx.network.add_plugin_v2([input_trt], plugin)
        output._trt = layer.get_output(0)


    @add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)])
    def test_example_layer_scale3():
        return ExampleLayer(3.0)


    @add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 4, 6)])
    def test_example_layer_scale4():
        return ExampleLayer(4.0)
except:
    pass # TODO: log plugin not found