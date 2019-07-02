import os
import glob
import shutil
from setuptools import setup, find_packages
from setuptools.command.install import install
from distutils.cmd import Command
from build import build

package_data = {}


class InstallCommand(install):
    description = "Builds the package"
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


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    PY_CLEAN_FILES = './build ./dist ./__pycache__ ./*.pyc ./*.tgz ./*.egg-info'.split(' ')
    description = "Command to tidy up the project root"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        root_dir = os.path.dirname(os.path.realpath(__file__))
        for path_spec in self.PY_CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(root_dir, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(root_dir):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, root_dir))
                print('removing %s' % os.path.relpath(path))
                shutil.rmtree(path)

        cmd_list = {
            "Removing generated protobuf cc files": "find . -name '*.pb.cc' -print0 | xargs -0 rm -f;",
            "Removing generated protobuf h files": "find . -name '*.pb.h' -print0 | xargs -0 rm -f;",
            "Removing generated protobuf py files": "find . -name '*_pb2.py' -print0 | xargs -0 rm -f;",
            "Removing generated ninja files": "find . -name '*.ninja*' -print0 | xargs -0 rm -f;"
        }

        for cmd, script in cmd_list.items():
            print("Running {}".format(cmd))
            os.system(script)


setup(
    name='torch2trt',
    version='0.0.0',
    description='An easy to use PyTorch to TensorRT converter',
    cmdclass={
        'install': InstallCommand,
        'clean': CleanCommand,
    },
    packages=find_packages(),
    package_data=package_data
)
