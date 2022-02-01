import tensorrt as trt
import numpy as np


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
            np.array([paddingLeft]).astype(np.int32),
            trt.PluginFieldType.INT32
        ),
        trt.PluginField(
            'paddingTop',
            np.array([paddingLeft]).astype(np.int32),
            trt.PluginFieldType.INT32
        ),
        trt.PluginField(
            'paddingBottom',
            np.array([paddingLeft]).astype(np.int32),
            trt.PluginFieldType.INT32
        )
    ])

    return creator.create_plugin('', fc)