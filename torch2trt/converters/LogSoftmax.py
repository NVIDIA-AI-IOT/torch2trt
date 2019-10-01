from torch2trt.torch2trt import *


@tensorrt_converter('torch.nn.LogSoftmax.forward')
def convert_LogSoftmax(ctx):
    input = ctx.method_args[1]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    layer = ctx.network.add_softmax(input=input_trt)
    layer = ctx.network.add_unary(input=layer.get_output(0),
            op=trt.UnaryOperation.LOG)
    output._trt = layer.get_output(0)