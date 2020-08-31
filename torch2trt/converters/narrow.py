import tensorrt as trt 
from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

@tensorrt_converter('torch.Tensor.narrow')
@tensorrt_converter('torch.narrow')
def convert_narrow(ctx):
    inputs = get_arg(ctx, 'input', pos=0, default=None)  
    start = get_arg(ctx, 'start', pos=2, default=None)
    output = ctx.method_return
    shape = list(inputs.shape)
    start = [0]*len(shape)
    stride = [1]*len(shape)
    dim = ctx.method_args[1] if get_arg(ctx, 'dim', pos=1, default=0) >=0 else len(shape)+get_arg(ctx, 'dim', pos=1, default=0)
    start[dim] = ctx.method_args[2]
    shape[dim] = ctx.method_args[3] 
    # not consider batch dimension
    input_trt = trt_(ctx.network,inputs)
    layer = ctx.network.add_slice(input=input_trt,start=start[1:], shape=shape[1:],stride=stride[1:])
    output._trt = layer.get_output(0)

class Narrow(torch.nn.Module):
    def __init__(self, dim, start, length):
        super(Narrow, self).__init__()
        self.dim = dim
        self.start = start
        self.length = length

    def forward(self, x):
        return torch.narrow(x,self.dim,self.start,self.length)

@add_module_test(torch.float32, torch.device('cuda'), [(1,3,224,224)])
def test_narrow1():
    return Narrow(1,0,2)

@add_module_test(torch.float32, torch.device('cuda'), [(1,3,224,224)])
def test_narrow2():
    return Narrow(2,0,50)


