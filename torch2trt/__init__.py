import warnings

from .torch2trt import *
from .converters import *
import tensorrt as trt


def load_plugins():
    import os
    import ctypes
    ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'libtorch2trt.so'))
    
    registry = trt.get_plugin_registry()
    torch2trt_creators = [c for c in registry.plugin_creator_list if c.plugin_namespace == 'torch2trt']
    for c in torch2trt_creators:
        registry.register_creator(c, 'torch2trt')


try:
    load_plugins()
    PLUGINS_LOADED = True
except OSError as e:
    warnings.warn("load_plugins OSError: {}, set PLUGINS_LOADED to False".format(e), RuntimeWarning)
    PLUGINS_LOADED = False
