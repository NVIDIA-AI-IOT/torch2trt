import sys
import argparse
from setuptools import setup, find_packages
from setuptools.command.install import install
from build import build

package_data = {}

class InstallCommand(install):
    description = "Builds plugins"
    user_options = install.user_options + [
        ('plugins', None, 'Build plugins'),
        ('cuda-dir=', None, 'Location of CUDA (if not default location)'),
        ('torch-dir=', None, 'Location of PyTorch (if not default location)'),
        ('trt-inc-dir=', None, 'Location of TensorRT include files (if not default location)'),
        ('trt-lib-dir=', None, 'Location of TensorRT libraries (if not default location)'),
    ]
    def initialize_options(self):
        install.initialize_options(self)
        self.plugins = False
        self.cuda_dir = None
        self.torch_dir = None
        self.trt_inc_dir = None
        self.trt_lib_dir = None
    def finalize_options(self):
        install.finalize_options(self)
    def run(self):
        if self.plugins:
            build_args = {}
            if self.cuda_dir:
                build_args['cuda_dir'] = self.cuda_dir
            if self.torch_dir:
                build_args['torch_dir'] = self.torch_dir
            if self.trt_inc_dir:
                build_args['trt_inc_dir'] = self.trt_inc_dir
            if self.trt_lib_dir:
                build_args['trt_lib_dir'] = self.trt_lib_dir

            print('Building in plugin support')
            build(**build_args)
            package_data['torch2trt'] = ['libtorch2trt.so']
        install.run(self)
            
setup(
    name='torch2trt',
    version='0.0.0',
    description='An easy to use PyTorch to TensorRT converter',
    cmdclass={
        'install': InstallCommand,
    },
    packages=find_packages(),
    package_data=package_data
)
