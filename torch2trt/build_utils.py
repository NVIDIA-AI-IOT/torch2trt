import imp
import os
from string import Template


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


PROTOC_CPP = \
"""
rule protoc_cpp
  command = protoc $in --cpp_out=. $flags
"""

PROTOC_PYTHON = \
"""
rule protoc_python
  command = protoc $in --python_out=. $flags
"""

CPP_OBJECT = \
"""
rule cpp_object
  command = g++ -c -fPIC -o $out $in $flags
"""

CPP_LIBRARY = \
"""
rule cpp_library
  command = g++ -shared -o $out $in $flags
"""

CPP_EXECUTABLE = \
"""
rule cpp_executable
  command = g++ -o $out $in $flags
"""


class Ninja(object):

    active_ninja = None

    def __init__(self):
        self.str = ''
        self.str += PROTOC_PYTHON
        self.str += PROTOC_CPP
        self.str += CPP_OBJECT
        self.str += CPP_LIBRARY
        self.str += CPP_EXECUTABLE

    def __enter__(self, *args, **kwargs):
        Ninja.active_ninja = self

    def __exit__(self, *args, **kwargs):
        Ninja.active_ninja = None

    def save(self, path='build.ninja'):
        with open(path, 'w') as f:
            f.write(self.str)


def filter_extensions(files, include_extensions=[], exclude_extensions=[]):
    return [f for f in files if (f.split('.')[-1] in include_extensions) and (f.split('.')[-1] not in exclude_extensions)]


def filter_cpp_sources(files):
    return filter_extensions(files, ['cxx', 'cpp', 'cc', 'c'])


def filter_cpp_headers(files):
    return filter_extensions(files, ['hpp', 'h'])


def filter_protos(files):
    return filter_extensions(files, ['proto'])


def protoc_cpp(srcs, include_dirs=[]):
    if isinstance(srcs, str):
        srcs = [srcs]
    srcs = filter_protos(srcs)
    outs = []
    for src in srcs:
        base = '.'.join(src.split('.')[:-1])
        cc = base + '.pb.cc'
        h = base + '.pb.h'
        outs += [cc, h]

    flags = ''
    flags += include_dir_string(include_dirs)

    ninja = Template(
"""
build $outs: protoc_cpp $srcs
  flags = $flags
"""
    ).substitute({
        'outs': ' '.join(outs),
        'srcs': ' '.join(srcs),
        'flags': flags
    })

    if Ninja.active_ninja is not None:
        Ninja.active_ninja.str += ninja

    return outs


def protoc_python(srcs, include_dirs=[]):
    if isinstance(srcs, str):
        srcs = [srcs]
    srcs = filter_protos(srcs)
    outs = []
    for src in srcs:
        base = '.'.join(src.split('.')[:-1])
        out = base + '_pb2.py'
        outs += [out]

    flags = ''
    flags += include_dir_string(include_dirs)

    ninja = Template(
"""
build $outs: protoc_python $srcs
  flags = $flags
"""
    ).substitute({
        'outs': ' '.join(outs),
        'srcs': ' '.join(srcs),
        'flags': flags
    })

    if Ninja.active_ninja is not None:
        Ninja.active_ninja.str += ninja

    return outs


def cpp_object(srcs, include_dirs=[], cflags=[]):
    if isinstance(srcs, str):
        srcs = [srcs]
    assert(len(srcs) == 1)
    src = srcs[0]
    base = '.'.join(src.split('.')[:-1])
    outs = [base + '.o']

    flags = ''
    flags += include_dir_string(include_dirs)
    flags += ' '.join(cflags)

    ninja = Template(
"""
build $outs: cpp_object $srcs
  flags = $flags
"""
    ).substitute({
        'outs': ' '.join(outs),
        'srcs': ' '.join(srcs),
        'flags': flags
    })

    if Ninja.active_ninja is not None:
        Ninja.active_ninja.str += ninja

    return outs


def cpp_library(out, srcs, include_dirs=[], library_dirs=[], libraries=[], cflags=[]):
    if isinstance(srcs, str):
        srcs = [srcs]
    srcs = filter_cpp_sources(srcs)
    assert(isinstance(out, str))
    outs = [out]

    objs = []
    for src in srcs:
        objs += cpp_object(src, include_dirs=include_dirs)

    flags = ''
    flags += library_dir_string(library_dirs)
    flags += library_string(libraries)
    flags += ' '.join(cflags)

    ninja = Template(
"""
build $outs: cpp_library $srcs
  flags = $flags
"""
    ).substitute({
        'outs': ' '.join(outs),
        'srcs': ' '.join(objs),
        'flags': flags
    })

    if Ninja.active_ninja is not None:
        Ninja.active_ninja.str += ninja

    return outs
