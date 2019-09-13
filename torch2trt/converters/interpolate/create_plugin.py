import torch
import os
from torch2trt.plugin import create_plugin


create_plugin(
    'interpolate',
    plugin_forward=
"""
auto input = input_tensors[0];
auto output = output_tensors[0];
if (msg.mode() == "bilinear") {
  at::upsample_bilinear2d_out(output, input, {msg.size(0), msg.size(1)}, msg.align_corners());
} else if (msg.mode() == "nearest") {
  at::upsample_nearest2d_out(output, input, {msg.size(0), msg.size(1)});
} else if (msg.mode() == "area") {
  at::adaptive_avg_pool2d_out(output, input, {msg.size(0), msg.size(1)});
} else if (msg.mode() == "bicubic") {
  at::upsample_bicubic2d_out(output, input, {msg.size(0), msg.size(1)}, msg.align_corners());
}
""",
    plugin_proto=
"""
repeated int64 size = 100;
string mode = 101;
bool align_corners = 102;
""",
    output_dir=os.path.dirname(os.path.abspath(__file__))
)