import sys
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

def trt_inc_dir():
    return "/usr/include/aarch64-linux-gnu"

def trt_lib_dir():
    return "/usr/lib/aarch64-linux-gnu"

ext_modules = []

plugins_ext_module = CUDAExtension(
        name='plugins', 
        sources=[
            'torch2trt/plugins/interpolate.cpp'
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
            'cxx': ['-DUSE_DEPRECATED_INTLIST'] if torch.__version__ < "1.5" else [],
            'nvcc': []
        }
)

if '--plugins' in sys.argv:
    ext_modules.append(plugins_ext_module)
    sys.argv.remove('--plugins')
    

setup(
    name='torch2trt',
    version='0.1.0',
    description='An easy to use PyTorch to TensorRT converter',
    packages=find_packages(),
    ext_package='torch2trt',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
