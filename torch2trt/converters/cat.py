from torch2trt.torch2trt import *


@tensorrt_converter('torch.cat')
def convert_cat(ctx):
    inputs = ctx.method_args[0]

    if 'dim' in ctx.method_kwargs:
        dim = ctx.method_kwargs['dim']
    else:
        dim = ctx.method_args[1]

    output = ctx.method_return
    trt_inputs = [trt_(ctx.network, i) for i in inputs]

    layer = ctx.network.add_concatenation(inputs=trt_inputs)
    layer.axis = dim - 1
    output._trt = layer.get_output(0)