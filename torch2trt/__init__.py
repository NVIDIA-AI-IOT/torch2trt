from .torch2trt import *
from .converters import *
import tensorrt as trt


def load_plugins():
    import torch2trt.plugins
    registry = trt.get_plugin_registry()
    torch2trt_creators = [c for c in registry.plugin_creator_list if c.plugin_namespace == 'torch2trt']
    for c in torch2trt_creators:
        registry.register_creator(c, 'torch2trt')


load_plugins()