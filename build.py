import imp
import subprocess
import os
from string import Template

PLUGINS = [
    'interpolate',
    'group_norm',
]

BASE_FOLDER = 'torch2trt/converters'

NINJA_TEMPLATE = Template((
    "rule link\n"
    "  command = g++ -shared -o $$out $$in -L$torch_dir/lib -L$cuda_dir/lib64 -L$trt_lib_dir -lc10 -lc10_cuda -ltorch -lcudart -lprotobuf -lprotobuf-lite -pthread -lpthread -lnvinfer\n"
    "rule protoc\n"
    "  command = protoc $$in --cpp_out=. --python_out=.\n"
    "rule cxx\n"
    "  command = g++ -c -fPIC $$in -I$cuda_dir/include -I$torch_dir/include -I$torch_dir/include/torch/csrc/api/include -I. -std=c++11 -I$trt_inc_dir\n"
))

PLUGIN_TEMPLATE = Template((
    "build $plugin_dir/$plugin.pb.h $plugin_dir/$plugin.pb.cc $plugin_dir/${plugin}_pb2.py: protoc $plugin_dir/$plugin.proto\n"
    "build $plugin.pb.o: cxx $plugin_dir/$plugin.pb.cc\n"
    "build $plugin.o: cxx $plugin_dir/$plugin.cpp\n"
))


def build(cuda_dir="/usr/local/cuda",
          torch_dir=imp.find_module('torch')[1],
          trt_inc_dir="/usr/include/aarch64-linux-gnu",
          trt_lib_dir="/usr/lib/aarch64-linux-gnu"):

    global PLUGINS, BASE_FOLDER, NINJA_TEMPLATE, PLUGIN_TEMPLATE

    NINJA_STR = NINJA_TEMPLATE.substitute({
        'torch_dir': torch_dir,
        'cuda_dir': cuda_dir,
        'trt_inc_dir': trt_inc_dir,
        'trt_lib_dir': trt_lib_dir,
    })


    plugin_o_files = []
    for plugin in PLUGINS:
        NINJA_STR += \
            PLUGIN_TEMPLATE.substitute({
                'plugin': plugin,
                'plugin_dir': os.path.join(BASE_FOLDER, plugin),
            })
        plugin_o_files += [plugin + '.pb.o', plugin + '.o']

    NINJA_STR += Template((
        "build torch2trt/libtorch2trt.so: link $o_files\n"
    )).substitute({'o_files': ' '.join(plugin_o_files)})

    with open('build.ninja', 'w') as f:
        f.write(NINJA_STR)

    subprocess.call(['ninja'])


if __name__ == '__main__':
    build()
