import tensorrt as trt
import numpy as np


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