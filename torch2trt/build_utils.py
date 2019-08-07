import imp
import subprocess
import os
from string import Template
import subprocess

PROTO_FILE_EXTENSIONS = ['proto']
CXX_FILE_EXTENSIONS = ['cxx', 'cpp', 'cc']

def find_torch_dir():
    return imp.find_module('torch')[1]


def find_cuda_dir():
    return '/usr/local/cuda'


def torch2trt_dep_libraries():
    libs = [
        'c10',
        'c10_cuda',
        'torch',
        'cudart',
        'caffe2',
        'caffe2_gpu',
        'protobuf',
        'protobuf-lite',
        'pthread',
        'nvinfer'
    ]
    return libs


def torch2trt_dep_include_dirs():
    dirs = [
        os.path.join(find_torch_dir(), 'include'),
        os.path.join(find_torch_dir(), 'include/torch/csrc/api/include'),
        os.path.join(find_cuda_dir(), 'include')
    ]
    return dirs


def torch2trt_dep_library_dirs():
    dirs = [
        os.path.join(find_torch_dir(), 'lib'),
        os.path.join(find_cuda_dir(), 'lib64')
    ]
    return dirs


def gcc_library_string(libs):
    s = ''
    for lib in libs:
        s += '-l' + lib + ' '
    return s


def gcc_library_dir_string(lib_dirs):
    s = ''
    for dir in lib_dirs:
        s += '-L' + dir + ' '
    return s


def gcc_include_dir_string(include_dirs):
    s = ''
    for dir in include_dirs:
        s += '-I' + dir + ' '
    return s

def _abspath_all(paths):
    return [os.path.abspath(p) for p in paths]

NINJA_PROTOC_PYTHON_RULE = \
"""
rule protoc_python
  command = cd $output_dir && protoc $in --python_out=. $proto_dirs
"""

def protoc_python_build(inputs, output_dir='.'):
    output_dir = os.path.abspath(output_dir)
    inputs = [input for input in inputs if input.split('.')[-1] in PROTO_FILE_EXTENSIONS]
    if isinstance(inputs, str):
        inputs = [inputs]
    
    outputs = []
    proto_dirs = []
    for input in inputs:
        input_basename = '.'.join(os.path.basename(input).split('.')[:-1])
        filename = input_basename + '_pb2.py'
        outputs += [os.path.join(output_dir, filename)]
        proto_dirs += [os.path.abspath(os.path.dirname(input))]
        
    inputs = _abspath_all(inputs)
    outputs = _abspath_all(outputs)
    
    ninja_str = Template(
"""
build $outputs: protoc_python $inputs
  output_dir = $output_dir
  proto_dirs = $proto_dirs
"""
    ).substitute({
        'inputs': ' '.join(inputs),
        'outputs': ' '.join(outputs),
        'output_dir': output_dir,
        'proto_dirs': gcc_include_dir_string(proto_dirs)
    })
    
    return ninja_str, outputs


NINJA_PROTOC_CPP_RULE = \
"""
rule protoc_cpp
  command = cd $output_dir && protoc $in --cpp_out=. $proto_dirs
"""

def protoc_cpp_build(inputs, output_dir='.'):
    output_dir = os.path.abspath(output_dir)
    inputs = [input for input in inputs if input.split('.')[-1] in PROTO_FILE_EXTENSIONS]
    if isinstance(inputs, str):
        inputs = [inputs]
    
    outputs = []
    proto_dirs = []
    for input in inputs:
        input_basename = '.'.join(os.path.basename(input).split('.')[:-1])
        cc_filename = input_basename + '.pb.cc'
        h_filename = input_basename + '.pb.h'
        outputs += [
            os.path.join(output_dir, cc_filename),
            os.path.join(output_dir, h_filename)
        ]
        proto_dirs += [os.path.abspath(os.path.dirname(input))]
        
    inputs = _abspath_all(inputs)
    outputs = _abspath_all(outputs)
    
    ninja_str = Template(
"""
build $outputs: protoc_cpp $inputs
  output_dir = $output_dir
  proto_dirs = $proto_dirs
"""
    ).substitute({
        'inputs': ' '.join(inputs),
        'outputs': ' '.join(outputs),
        'output_dir': output_dir,
        'proto_dirs': gcc_include_dir_string(proto_dirs)
    })
    
    return ninja_str, outputs


NINJA_CPP_COMPILE_RULE = \
"""
compiler = g++
rule cpp_compile
  command = cd $output_dir && $compiler -c -fPIC $in $include_dirs
"""

def cpp_compile_build(inputs, output_dir='.', include_dirs=[], compiler='g++'):
    output_dir = os.path.abspath(output_dir)
    inputs = [input for input in inputs if input.split('.')[-1] in CXX_FILE_EXTENSIONS]
    if isinstance(inputs, str):
        inputs = [inputs]
        
    outputs = []
    for input in inputs:
        input_basename = '.'.join(os.path.basename(input).split('.')[:-1])
        o_filename = input_basename + '.o'
        outputs += [
            os.path.join(output_dir, o_filename),
        ]
        
    inputs = _abspath_all(inputs)
    outputs = _abspath_all(outputs)
    include_dirs = _abspath_all(include_dirs)
        
    ninja_str = Template(
"""
build $outputs: cpp_compile $inputs
  compiler = $compiler
  output_dir = $output_dir
  include_dirs = $include_dirs
"""
    ).substitute({
        'inputs': ' '.join(inputs),
        'outputs': ' '.join(outputs),
        'compiler': compiler,
        'output_dir': output_dir,
        'include_dirs': gcc_include_dir_string(include_dirs)
    })
    
    return ninja_str, outputs


NINJA_CPP_LINK_RULE = \
"""
compiler = g++
rule cpp_link
  command = $compiler -shared -o $out $in $lib_dirs $libs
"""

def cpp_link_build(name, inputs, output_dir='.', libs=[], lib_dirs=[], compiler='g++'):
    output_dir = os.path.abspath(output_dir)
    inputs = [input for input in inputs if input.split('.')[-1] in ['o']]
    if isinstance(inputs, str):
        inputs = [inputs]
    
    inputs = _abspath_all(inputs)
    output = os.path.abspath(os.path.join(output_dir, 'lib' + name + '.so'))
    
    ninja_str = Template(
"""
build $output: cpp_link $inputs
  compiler = $compiler
  libs = $libs
  lib_dirs = $lib_dirs
"""
    ).substitute({
        'compiler': compiler,
        'inputs': ' '.join(inputs),
        'output': output,
        'libs': gcc_library_string(libs),
        'lib_dirs': gcc_library_dir_string(lib_dirs)
    })
    
    return ninja_str, [output]


def _ninja_build_library(name, srcs, protos, include_dirs=[], library_dirs=[], libraries=[]):
    NINJA_STR = ''
    NINJA_STR += NINJA_PROTOC_PYTHON_RULE
    NINJA_STR += NINJA_PROTOC_CPP_RULE
    NINJA_STR += NINJA_CPP_COMPILE_RULE
    NINJA_STR += NINJA_CPP_LINK_RULE
    
    objects = []
    for proto in protos:
        ninja_str, _ = protoc_python_build([proto], output_dir=os.path.dirname(proto))
        NINJA_STR += ninja_str
        ninja_str, outputs = protoc_cpp_build([proto], output_dir=os.path.dirname(proto))
        NINJA_STR += ninja_str
        ninja_str, outputs = cpp_compile_build(outputs, output_dir=os.path.dirname(proto), include_dirs=include_dirs)
        NINJA_STR += ninja_str
        objects += outputs
        
    for src in srcs:
        ninja_str, outputs = cpp_compile_build([src], output_dir=os.path.dirname(src), include_dirs=include_dirs)
        NINJA_STR += ninja_str
        objects += outputs
        
    ninja_str, outputs = cpp_link_build(name, objects, libs=libraries, lib_dirs=library_dirs)
    NINJA_STR += ninja_str
    return NINJA_STR, outputs


def build_library(name, srcs, protos, include_dirs=[], library_dirs=[], libraries=[]):
    with open('build.ninja', 'w') as f:
        ninja_str, _ = _ninja_build_library(name, srcs, protos, include_dirs, library_dirs, libraries)
        f.write(ninja_str)
    subprocess.call(['ninja'])