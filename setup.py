import sys
from setuptools import setup, find_packages

def trt_inc_dir():
    return "/usr/include/aarch64-linux-gnu"

def trt_lib_dir():
    return "/usr/lib/aarch64-linux-gnu"

ext_modules = []

if '--plugins' in sys.argv:
    import torch
    from torch.utils.cpp_extension import CUDAExtension
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
                'cxx': ['-DUSE_DEPRECATED_INTLIST'] if torch.__version__ < "1.5" else [],
                'nvcc': []
            }
        )
    ext_modules.append(plugins_ext_module)
    sys.argv.remove('--plugins')


def get_cmdclass():
    if '--plugins' in sys.argv:
        from torch.utils.cpp_extension import BuildExtension
        return {'build_ext': BuildExtension}
    else:
        return {}


setup(
    name='torch2trt',
    version='0.1.0',
    description='An easy to use PyTorch to TensorRT converter',
    packages=find_packages(),
    ext_package='torch2trt',
    ext_modules=ext_modules,
    cmdclass=get_cmdclass()
)

