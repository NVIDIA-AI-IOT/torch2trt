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
         
        
def create_add_plugin_method(plugin_name, output_dir='.'):
    TEMPLATE = \
"""
import torch
import os
import tensorrt as trt
from torch2trt.plugin import load_plugin_library
from ${PLUGIN_NAME}_pb2 import ${PLUGIN_NAME}_Msg

library_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'torch2trt_plugin_${PLUGIN_NAME}.so')
load_plugin_library(library_path)

def add_${PLUGIN_NAME}(network, inputs, outputs, **kwargs):
    
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
    tmp = Template(TEMPLATE).substitute({'PLUGIN_NAME': plugin_name})
    with open(os.path.join(output_dir, 'add_' + plugin_name + '.py'), 'w') as f:
        f.write(tmp)

        
def create_plugin(plugin_name, plugin_members="", plugin_setup="", plugin_forward="", plugin_proto="", extra_src="", output_dir='.'):
    
    library_name = 'torch2trt_plugin_' + plugin_name + '.so'
    library_path = os.path.join(output_dir, library_name)

    src = Template(PLUGIN_SRC_TEMPLATE).substitute({
        'EXTRA_SRC': extra_src,
        'PLUGIN_NAME': plugin_name,
        'PLUGIN_MEMBERS': plugin_members,
        'PLUGIN_SETUP': plugin_setup,
        'PLUGIN_FORWARD': plugin_forward,
    })

    proto = Template(PLUGIN_PROTO_TEMPLATE).substitute({
        'PLUGIN_NAME': plugin_name,
        'PLUGIN_PROTO': plugin_proto
    })

    flags = ''
    flags += include_dir_string(include_dirs())
    flags += library_dir_string(library_dirs())
    flags += library_string(libraries())

    ninja = Template(PLUGIN_NINJA_TEMPLATE).substitute({
        'PLUGIN_NAME': plugin_name,
        'FLAGS': flags
    })
    
    plugin_output_dir = os.path.join(output_dir, )
        
    with open(os.path.join(output_dir, plugin_name + '.cu'), 'w') as f:
        f.write(src)
    
    with open(os.path.join(output_dir, plugin_name + '.proto'), 'w') as f:
        f.write(proto)
        
    with open(os.path.join(output_dir, 'build.ninja'), 'w') as f:
        f.write(ninja)
        
    create_add_plugin_method(plugin_name, output_dir=output_dir)
    subprocess.call(['ninja'], cwd=output_dir)