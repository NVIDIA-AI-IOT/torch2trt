from torch2trt.torch2trt import *


@tensorrt_converter('torch.Tensor.__iadd__')
@tensorrt_converter('torch.Tensor.__add__')
def convert_add(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a._trt, input_b._trt, trt.ElementWiseOperation.SUM)
    output._trt = layer.get_output(0)