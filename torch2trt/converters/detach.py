from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter("torch.Tensor.detach")
def convert_detach(ctx):
    input = get_arg(ctx, "self", 0, None)
    output = ctx.method_return
    output._trt = input._trt