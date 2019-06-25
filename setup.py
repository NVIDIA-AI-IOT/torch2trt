from setuptools import setup, find_packages
from build import build

try:
    build()
except e:
    print('Could not build plugins')

setup(
    name='torch2trt',
    version='0.0',
    description='PyTorch to TensorRT converter',
    packages=find_packages(),
    package_data={'torch2trt': ['libtorch2trt.so']}
)
