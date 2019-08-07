import imp
import subprocess
import os
from string import Template
import sys
sys.path.append('torch2trt')
import torch2trt.build_utils

def build():
    torch2trt.build_utils.build_library(
        name='torch2trt_plugins', 
        srcs=['torch2trt/plugins/torch_plugin.cpp'], 
        protos=['torch2trt/plugins/torch_plugin.proto'], 
        include_dirs=torch2trt.build_utils.torch2trt_dep_include_dirs(), 
        library_dirs=torch2trt.build_utils.torch2trt_dep_library_dirs(), 
        libraries=torch2trt.build_utils.torch2trt_dep_libraries()
    )
    
if __name__ == '__main__':
    build()