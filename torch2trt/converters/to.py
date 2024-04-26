from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter("torch.Tensor.to")
def convert_to(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    output = ctx.method_return
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output._trt = input_trt