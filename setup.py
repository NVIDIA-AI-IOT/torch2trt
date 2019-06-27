import sys
from setuptools import setup, find_packages
from build import build

package_data = {}

if '--plugins' in sys.argv:
    sys.argv.remove('--plugins')
    try:
        build()
        package_data['torch2trt'] = ['libtorch2trt.so']
    except RuntimeError:
        print('Could not build plugins')

setup(
    name='torch2trt',
    version='0.0.0',
    description='An easy to use PyTorch to TensorRT converter',
    packages=find_packages(),
    package_data=package_data
)
