import imp
import subprocess
import os
from string import Template

PLUGINS = [
    'interpolate',
]

BASE_FOLDER = 'torch2trt/converters'

NINJA_STR = Template(
"""
rule link
  command = g++ -shared -o $$out $$in -L$torch_dir/lib -L$cuda_dir/lib64 -lc10 -lc10_cuda -ltorch -lcudart -lcaffe2 -lcaffe2_gpu -lprotobuf -lprotobuf-lite -pthread -lpthread -lnvinfer

rule protoc
  command = protoc $$in --cpp_out=. --python_out=.

rule cxx
  command = g++ -c -fPIC $$in -I$cuda_dir/include -I$torch_dir/include -I$torch_dir/include/torch/csrc/api/include -I. 

"""
).substitute({
    'torch_dir': imp.find_module('torch')[1],
    'cuda_dir': '/usr/local/cuda'
})

PLUGIN_TEMPLATE = Template(
"""
build $plugin_dir/$plugin.pb.h $plugin_dir/$plugin.pb.cc $plugin_dir/${plugin}_pb2.py: protoc $plugin_dir/$plugin.proto
build $plugin.pb.o: cxx $plugin_dir/$plugin.pb.cc
build $plugin.o: cxx $plugin_dir/$plugin.cpp
"""
)


def build():
    global PLUGINS, BASE_FOLDER, NINJA_STR, PLUGIN_TEMPLATE
    plugin_o_files = []
    for plugin in PLUGINS:
        NINJA_STR += \
            PLUGIN_TEMPLATE.substitute({
                'plugin': plugin,
                'plugin_dir': os.path.join(BASE_FOLDER, plugin),
            })
        plugin_o_files += [plugin + '.pb.o', plugin + '.o']

    NINJA_STR += Template(
"""
build torch2trt/libtorch2trt.so: link $o_files
"""
    ).substitute({'o_files': ' '.join(plugin_o_files)})

    with open('build.ninja', 'w') as f:
        f.write(NINJA_STR)

    subprocess.call(['ninja'])

if __name__ == '__main__':
    build()
