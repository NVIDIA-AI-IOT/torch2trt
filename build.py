import subprocess
import torch2trt.build_utils as bt


def build():
    ninja = bt.Ninja()

    with ninja:

        generated = bt.protoc_cpp(
            srcs=[
                'torch2trt/plugins/interpolate_plugin.proto',
                'torch2trt/plugins/torch_plugin.proto',
            ],
            include_dirs=['.']
        )

        library = bt.cpp_library(
            out='torch2trt/libtorch2trt_plugins.so',
            srcs=generated + ['torch2trt/plugins/interpolate_plugin.cpp'],
            include_dirs=['.'] + bt.include_dirs(),
            library_dirs=bt.library_dirs(),
            libraries=bt.libraries()
        )

    ninja.save()
    subprocess.call(['ninja'])


if __name__ == '__main__':
    build()
