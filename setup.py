import sys
from setuptools import setup, find_packages
from torch2trt.plugin import build_plugins

package_data = {}

if '--plugins' in sys.argv:
    sys.argv.remove('--plugins')
    build_plugins()

setup(
    name='torch2trt',
    version='0.0.0',
    description='An easy to use PyTorch to TensorRT converter',
    packages=find_packages(),
    package_data=package_data
)
