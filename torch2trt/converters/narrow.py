import tensorrt as trt 
from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.narrow')
def convert_narrow(ctx):
    inputs = get_arg(ctx, 'input', pos=0, default=None) 
    dim = get_arg(ctx, 'dim', pos=1, default=0) 
    start = get_arg(ctx, 'start', pos=2, default=None)
    print("start",start, type(start))
    input_trt= trt_(ctx.network, inputs)
    
    output = ctx.method_return
    output_shape = list(output.size())
    #print(type(trt.Dims(start)),type(trt.Tensorrt.Dims(start)))
    print(type(trt.tensorrt.Dims(output_shape)))
    layer = ctx.network.add_slice(inputs=input_trt,shape=trt.tensorrt.Dims(output_shape))
    output._trt = layer.get_output(0)

class narrow(torch.nn.Module):
    def __init__(self, dim, start, length):
        super(Cat, self).__init__()
        self.dim = dim
        self.start = start
        self.length = length

    def forward(self, *x):
        return torch.narrow(x, dim=self.dim,start=self.start,length=self.length)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 3, 4), (1, 17, 4)])
def test_narrow_basic():
    return narrow(1,0,2)
