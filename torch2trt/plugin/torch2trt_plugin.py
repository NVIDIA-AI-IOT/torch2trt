import imp
import os
from string import Template
from ninja_template import PLUGIN_NINJA_TEMPLATE
from proto_template import PLUGIN_PROTO_TEMPLATE
from src_template import PLUGIN_SRC_TEMPLATE


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


def torch2trt_plugin(plugin_name, plugin_members="", plugin_setup="", plugin_forward="", plugin_proto="", extra_src=""):
    
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
        
    with open(plugin_name + '.cu', 'w') as f:
        f.write(src)
    
    with open(plugin_name + '.proto', 'w') as f:
        f.write(proto)
        
    with open('build.ninja', 'w') as f:
        f.write(ninja)