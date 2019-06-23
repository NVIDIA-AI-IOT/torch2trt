from torch2trt.torch2trt import *


@tensorrt_converter('torch.Tensor.view')
def convert_view(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return
    layer = ctx.network.add_shuffle(input._trt)
    layer.reshape_dims = tuple(output.shape[1:])
    output._trt = layer.get_output(0)
