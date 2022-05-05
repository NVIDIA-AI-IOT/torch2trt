import os
import ctypes

HAS_PLUGINS = False

try:
    handle = ctypes.CDLL('libtorch2trt_plugins.so')
    HAS_PLUGINS = True
    from .converters import *
    from .creators import *
except:
    HAS_PLUGINS = False

def has_plugins():
    return HAS_PLUGINS