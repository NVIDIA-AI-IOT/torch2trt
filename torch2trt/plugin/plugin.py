import imp
import os
import ctypes
import subprocess
import tensorrt as trt
from string import Template
from .ninja_template import PLUGIN_NINJA_TEMPLATE
from .proto_template import PLUGIN_PROTO_TEMPLATE
from .src_template import PLUGIN_SRC_TEMPLATE


def find_torch_dir():
    return imp.find_module('torch')[1]


def find_cuda_dir():
    return '/usr/local/cuda'


def libraries():
    libs = [
        'c10',
        'c10_cuda',
        'torch',
        'cudart',
        'protobuf',
        'protobuf-lite',
        'pthread',
        'nvinfer'
    ]
    return libs


def include_dirs():
    dirs = [
        os.path.join(find_torch_dir(), 'include'),
        os.path.join(find_torch_dir(), 'include/torch/csrc/api/include'),
        os.path.join(find_cuda_dir(), 'include')
    ]
    return dirs


def library_dirs():
    dirs = [
        os.path.join(find_torch_dir(), 'lib'),
        os.path.join(find_cuda_dir(), 'lib64')
    ]
    return dirs


def include_dir_string(include_dirs):
    s = ''
    for d in include_dirs:
        s += '-I' + d + ' '
    return s


def library_string(libraries):
    s = ''
    for l in libraries:
        s += '-l' + l + ' '
    return s


def library_dir_string(library_dirs):
    s = ''
    for l in library_dirs:
        s += '-L' + l + ' '
    return s


def abspath(paths):
    if isinstance(paths, str):
        return os.path.abspath(paths)
    return [os.path.abspath(p) for p in paths]


def refresh_plugin_registry():
    registry = trt.get_plugin_registry()
    torch2trt_creators = [c for c in registry.plugin_creator_list if c.plugin_namespace == 'torch2trt']
    for c in torch2trt_creators:
        registry.register_creator(c, 'torch2trt')
        
        
def load_plugin_library(path):
    ctypes.CDLL(path)
    refresh_plugin_registry()
    

class Plugin(object):
    
    def __init__(self, name, members='', setup='', forward='', extra_src='', proto='', directory='~/.torch2trt', extra_include_dirs=[], extra_library_dirs=[], extra_libraries=[], cflags=[]):
        self.name = name
        self.members = members
        self.setup = setup
        self.forward = forward
        self.extra_src = extra_src
        self.proto = proto
        self.directory = os.path.abspath(os.path.expanduser(directory))
        self.include_dirs = include_dirs() + extra_include_dirs
        self.library_dirs = library_dirs() + extra_library_dirs
        self.libraries = libraries() + extra_libraries
        self.cflags = cflags
        
    def _src_str(self):
        src = Template(PLUGIN_SRC_TEMPLATE).substitute({
            'EXTRA_SRC': self.extra_src,
            'PLUGIN_NAME': self.name,
            'PLUGIN_MEMBERS': self.members,
            'PLUGIN_SETUP': self.setup,
            'PLUGIN_FORWARD': self.forward,
        })
        return src
    
    def _proto_str(self):
        proto = Template(PLUGIN_PROTO_TEMPLATE).substitute({
            'PLUGIN_NAME': self.name,
            'PLUGIN_PROTO': self.proto
        })
        return proto
    
    def _ninja_str(self):
        flags = ''
        flags += include_dir_string(self.include_dirs)
        flags += library_dir_string(self.library_dirs)
        flags += library_string(self.libraries)
        flags += ' '.join(self.cflags)

        ninja = Template(PLUGIN_NINJA_TEMPLATE).substitute({
            'PLUGIN_NAME': self.name,
            'PLUGIN_LIB_NAME': self._lib_name(),
            'FLAGS': flags
        })
        return ninja
    
    def _lib_name(self):
        return 'libtorch2trt_plugin_' + self.name + '.so'
    
    def _lib_path(self):
        return os.path.join(self.directory, self._lib_name())
    
    def _load(self):
        load_plugin_library(self._lib_path())
        
    def build(self):
        try:
            self._load()
        except:
            print('%s plugin not built, building it now...' % self.name)
            self.rebuild()
            self._load()
        
    def rebuild(self):
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
            
        src_path = os.path.join(self.directory, self.name + '.cu')
        proto_path = os.path.join(self.directory, self.name + '.proto')
        ninja_path = os.path.join(self.directory, 'build.ninja')
        
        with open(src_path, 'w') as f:
            f.write(self._src_str())
            
        with open(proto_path, 'w') as f:
            f.write(self._proto_str())
        
        with open(ninja_path, 'w') as f:
            f.write(self._ninja_str())
            
        subprocess.call(['ninja'], cwd=self.directory)
        
    def add_to_network(self, network, inputs, outputs, **kwargs):
        import sys
        sys.path.append(self.directory)
        _locals = locals()
        TEMPLATE = \
"""
try:
    from .${PLUGIN_NAME}_pb2 import ${PLUGIN_NAME}_Msg
except:
    from ${PLUGIN_NAME}_pb2 import ${PLUGIN_NAME}_Msg

library_path = "${PLUGIN_LIB_PATH}"
load_plugin_library(library_path)
    
msg = ${PLUGIN_NAME}_Msg(**kwargs)
    
for input in inputs:
    msg.input_shapes.add(size=tuple(input.shape[1:]))
for output in outputs:
    msg.output_shapes.add(size=tuple(output.shape[1:]))
    
registry = trt.get_plugin_registry()
creator = registry.get_plugin_creator("${PLUGIN_NAME}", '1', 'torch2trt')
plugin = creator.deserialize_plugin("${PLUGIN_NAME}", msg.SerializeToString())
    
layer = network.add_plugin_v2([input._trt for input in inputs], plugin)
for i in range(len(outputs)):
    outputs[i]._trt = layer.get_output(i)
"""
        tmp = Template(TEMPLATE).substitute({
            'PLUGIN_NAME': self.name,
            'PLUGIN_LIB_PATH': self._lib_path()
        })

        exec(tmp, globals(), _locals)