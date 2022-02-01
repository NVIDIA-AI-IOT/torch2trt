import os
import ctypes


def plugin_library_path():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'libtorch2trt_plugins.so')
    return filename
    

def has_plugins():
    return os.path.exists(plugin_library_path())


def load_plugins():
    if not has_plugins():
        return False
    else:
        ctypes.CDLL(plugin_library_path())
        return True