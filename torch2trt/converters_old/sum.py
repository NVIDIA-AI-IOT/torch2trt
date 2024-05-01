from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from .unary import UnaryModule
from torch import nn

@tensorrt_converter('torch.sum')
@tensorrt_converter('torch.Tensor.sum')
def convert_sum(ctx):
    input = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=tuple(range(1, len(input.shape))))
    keepdim = get_arg(ctx, 'keepdim', pos=2, default=False)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    layer = ctx.network.add_reduce(input_trt,  trt.ReduceOperation.SUM, torch_dim_to_trt_axes(dim), keepdim)
    output._trt = layer.get_output(0)
        
        
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_sum_reduce_all():
    return UnaryModule(lambda x: torch.sum(x))     


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_sum_reduce_dim1():
    return UnaryModule(lambda x: torch.sum(x, 1))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_sum_reduce_dim22():
    return UnaryModule(lambda x: torch.sum(x, 2))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_sum_reduce_dim1_keepdim():
    return UnaryModule(lambda x: torch.sum(x, 1, keepdim=True))


class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.register_buffer('disp', torch.arange(maxdisp, dtype=torch.float32).view(maxdisp, 1, 1))

    def forward(self, x):      
        return x * self.disp#, 1) 

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 23, 23)])
def test_disparity_reg():
    return DisparityRegression(10)    
