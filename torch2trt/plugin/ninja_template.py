PLUGIN_NINJA_TEMPLATE = \
"""
rule protoc_cpp
  command = protoc $$in --cpp_out=.

rule protoc_python
  command = protoc $$in --python_out=.

rule cuda_library
  command = nvcc -shared --compiler-options '-fPIC' -o $$out $$in $$flags


build ${PLUGIN_NAME}_pb2.py: protoc_python ${PLUGIN_NAME}.proto

build ${PLUGIN_NAME}.pb.h ${PLUGIN_NAME}.pb.cc: protoc_cpp ${PLUGIN_NAME}.proto

build torch2trt_plugin_${PLUGIN_NAME}.so: cuda_library ${PLUGIN_NAME}.cu ${PLUGIN_NAME}.pb.cc
  flags = ${FLAGS}

"""