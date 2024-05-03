import sys
import tensorrt
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from packaging import version


def trt_inc_dir():
    return "/usr/include/aarch64-linux-gnu"

def trt_lib_dir():
    return "/usr/lib/aarch64-linux-gnu"

ext_modules = []
exclude_dir = ["torch2trt/contrib","torch2trt/contrib.*"]

compile_args_cxx = []
if version.parse(torch.__version__) < version.parse('1.5'):
    compile_args_cxx.append('-DUSE_DEPRECATED_INTLIST')
if version.parse(tensorrt.__version__) < version.parse('8'):
    compile_args_cxx.append('-DPRE_TRT8')

plugins_ext_module = CUDAExtension(
        name='plugins',
        sources=[
            'torch2trt/plugins/plugins.cpp'
        ],
        include_dirs=[
            trt_inc_dir()
        ],
        library_dirs=[
            trt_lib_dir()
        ],
        libraries=[
            'nvinfer'
        ],
        extra_compile_args={
            'cxx': compile_args_cxx,
            'nvcc': []
        }
    )

if '--plugins' in sys.argv:
    ext_modules.append(plugins_ext_module)
    sys.argv.remove('--plugins')

if '--contrib' in sys.argv:
    exclude_dir=[]
    sys.argv.remove('--contrib')

setup(
    name='torch2trt',
    version='0.5.0',
    description='An easy to use PyTorch to TensorRT converter',
    packages=find_packages(exclude=exclude_dir),
    ext_package='torch2trt',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
