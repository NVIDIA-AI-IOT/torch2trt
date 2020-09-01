from torch2trt.torch2trt import *


@tensorrt_converter('torch.nn.Dropout.forward')
@tensorrt_converter('torch.nn.Dropout2d.forward')
@tensorrt_converter('torch.nn.Dropout3d.forward')
def convert_Identity(ctx):
    input = ctx.method_args[1]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    output._trt = input_trt