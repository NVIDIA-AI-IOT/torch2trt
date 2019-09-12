PLUGIN_PROTO_TEMPLATE = \
"""
syntax = "proto3";

package torch2trt.${PLUGIN_NAME};

enum DataTypeMsg {
  kFloat = 0;
  kHalf = 1;
  kInt8 = 2;
  kInt32 = 3;
}

message TensorShapeMsg {
  repeated int64 size = 1; // does not include batch
}

message ${PLUGIN_NAME}_Msg {
  DataTypeMsg dtype = 1;
  repeated TensorShapeMsg input_shapes = 2;
  repeated TensorShapeMsg output_shapes = 3;
  
  ${PLUGIN_PROTO}
}
"""