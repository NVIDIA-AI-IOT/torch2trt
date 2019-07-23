from torch2trt.torch2trt import *

@tensorrt_converter('torch.Tensor.narrow')
def convert_narrow(ctx):
    input_a = ctx.method_args[0] 
    output = ctx.method_return
    assert len(ctx.method_args)==4, "args: [input, dim, start, length]!"
    shape = list(input_a.shape)
    start = [0]*len(shape)
    stride = [1]*len(shape)
    dim = ctx.method_args[1] if ctx.method_args[1]>=0 else len(shape)+ctx.method_args[1]
    start[dim] = ctx.method_args[2]
    shape[dim] = ctx.method_args[3] 
    # not consider batch dimension
    layer = ctx.network.add_slice(input=input_a._trt,start=start[1:], shape=shape[1:],stride=stride[1:])
    output._trt = layer.get_output(0)
