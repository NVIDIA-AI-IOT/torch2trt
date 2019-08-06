import imp
import subprocess
import os
from string import Template

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
        os.path.join(find_cuda_dir(), 'lib64')
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
        s += '-l' + lib
    return s


def gcc_library_dir_string(lib_dirs):
    s = ''
    for dir in lib_dirs:
        s += '-L' + dir + ' '
    return s


def gcc_include_dir_string(include_dirs):
    s = ''
    for dir in include_dirs:
        s += '-I' + dir
    return s


NINJA_PROTOC_PYTHON_RULE = \
"""
rule protoc_python
  command = protoc $in --python_out=. && mv $original_outputs $output_dir
"""

def protoc_python_build(inputs, output_dir='.'):
    inputs = [input for input in inputs if input.split('.')[-1] in PROTO_FILE_EXTENSIONS]
    if isinstance(inputs, str):
        inputs = [inputs]
    
    outputs = []
    original_outputs = []
    for input in inputs:
        input_dir = os.path.dirname(input)
        input_basename = '.'.join(os.path.basename(input).split('.')[:-1])
        filename = input_basename + '_pb2.py'
        outputs += [os.path.join(output_dir, filename)]
        original_outputs += [os.path.join(input_dir, filename)]
    
    ninja_str = Template(
"""
build $outputs: protoc_python $inputs
  original_outputs = $original_outputs
  output_dir = $output_dir
"""
    ).substitute({
        'inputs': ' '.join(inputs),
        'outputs': ' '.join(outputs),
        'original_outputs': ' '.join(original_outputs),
        'output_dir': output_dir
    })
    
    return ninja_str, outputs


NINJA_PROTOC_CPP_RULE = \
"""
rule protoc_cpp
  command = protoc $in --cpp_out=. && mv $original_outputs $output_dir
"""

def protoc_cpp_build(inputs, output_dir='.'):
    inputs = [input for input in inputs if input.split('.')[-1] in PROTO_FILE_EXTENSIONS]
    if isinstance(inputs, str):
        inputs = [inputs]
    
    outputs = []
    original_outputs = []
    for input in inputs:
        input_dir = os.path.dirname(input)
        input_basename = '.'.join(os.path.basename(input).split('.')[:-1])
        cc_filename = input_basename + '.pb.cc'
        h_filename = input_basename + '.pb.h'
        outputs += [
            os.path.join(output_dir, cc_filename),
            os.path.join(output_dir, h_filename)
        ]
        original_outputs += [
            os.path.join(input_dir, cc_filename),
            os.path.join(input_dir, h_filename)
        ]
    
    ninja_str = Template(
"""
build $outputs: protoc_cpp $inputs
  original_outputs = $original_outputs
  output_dir = $output_dir
"""
    ).substitute({
        'inputs': ' '.join(inputs),
        'outputs': ' '.join(outputs),
        'original_outputs': ' '.join(original_outputs),
        'output_dir': output_dir
    })
    
    return ninja_str, outputs


NINJA_CPP_COMPILE_RULE = \
"""
compiler = g++
rule cpp_compile
  command = $compiler -c -fPIC $in $include_dirs && mv $original_outputs $output_dir
"""

def cpp_compile_build(inputs, output_dir='.', include_dirs=[], compiler='g++'):
    inputs = [input for input in inputs if input.split('.')[-1] in CXX_FILE_EXTENSIONS]
    if isinstance(inputs, str):
        inputs = [inputs]
        
    outputs = []
    original_outputs = []
    for input in inputs:
        input_dir = os.path.dirname(input)
        input_basename = '.'.join(os.path.basename(input).split('.')[:-1])
        o_filename = input_basename + '.o'
        outputs += [
            os.path.join(output_dir, o_filename),
        ]
        original_outputs += [
            os.path.join(o_filename),
        ]
        
    ninja_str = Template(
"""
build $outputs: cpp_compile $inputs
  compiler = $compiler
  original_outputs = $original_outputs
  output_dir = $output_dir
"""
    ).substitute({
        'compiler': compiler,
        'inputs': ' '.join(inputs),
        'outputs': ' '.join(outputs),
        'original_outputs': ' '.join(original_outputs),
        'output_dir': output_dir
    })
    
    return ninja_str, outputs