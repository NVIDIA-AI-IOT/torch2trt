from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


def unsqueeze(ctx, input, dim):
    layer = ctx.network.add_shuffle(trt_(ctx.network, input))
  
    shape = input.shape[1:dim] + (1,) + input.shape[dim:]
    layer.reshape_dims = tuple(shape)

    return layer.get_output(0)


@tensorrt_converter('torch.stack', enabled=trt_version() >= '7.0')
def convert_cat_trt7(ctx):
    inputs = get_arg(ctx, 'input', pos=0, default=None) 
    dim = get_arg(ctx, 'dim', pos=1, default=0) 

    output = ctx.method_return
    trt_inputs = [unsqueeze(ctx, i, dim) for i in inputs]

    layer = ctx.network.add_concatenation(inputs=trt_inputs)
    layer.axis = dim - 1
    output._trt = layer.get_output(0)

class Stack(torch.nn.Module):
    def __init__(self, dim):
        super(Stack, self).__init__()
        self.dim = dim

    def forward(self, *x):
        return torch.stack(x, dim=self.dim)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 4, 4), (1, 4, 4)], enabled=trt_version() >= '7.0')
def test_Stack_basic_trt7():
    return Stack(3)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 4, 4), (1, 4, 4)], enabled=trt_version() >= '7.0')
def test_Stack_basic2_trt7():
    return Stack(1)
