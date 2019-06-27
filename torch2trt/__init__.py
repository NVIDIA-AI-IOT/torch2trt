from .torch2trt import *
from .converters import *


def load_plugins():
    import os
    import ctypes
    ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'libtorch2trt.so'))


try:
    load_plugins()
    PLUGINS_LOADED = True
except OSError:
    PLUGINS_LOADED = False
