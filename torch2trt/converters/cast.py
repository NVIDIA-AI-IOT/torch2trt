from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter("torch.Tensor.type")
@tensorrt_converter("torch.Tensor.float")
@tensorrt_converter("torch.Tensor.half")
@tensorrt_converter("torch.Tensor.int")
def convert_type(ctx):
    input = get_arg(ctx, "input", 0, None)
    output = ctx.method_return

    input_dtype = torch_dtype_to_trt(input.dtype)
    output_dtype = torch_dtype_to_trt(output.dtype)

    if input_dtype != output_dtype:
        layer = ctx.network.add_cast(input._trt, output_dtype)
        output._trt = layer.get_output(0)
    else:
        output._trt = input._trt


    