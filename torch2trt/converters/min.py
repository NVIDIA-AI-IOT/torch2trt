from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from .unary import UnaryModule


def __convert_min_elementwise(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.MIN)
    output._trt = layer.get_output(0)
    

def __convert_min_reduce(ctx):
    input = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=tuple(range(1,input.ndim)))
    keepdim = get_arg(ctx, 'keepdim', pos=2, default=False)
    input_trt= trt_(ctx.network, input)
    output_val = ctx.method_return[0]
    output_idx = ctx.method_return[1]
    layer = ctx.network.add_reduce(input_trt,  trt.ReduceOperation.MIN, torch_dim_to_trt_axes(dim), keepdim)
    output_val._trt = layer.get_output(0)
    

@tensorrt_converter('torch.min')
@tensorrt_converter('torch.Tensor.min')
def convert_min(ctx):
    if len(ctx.method_args) > 1 and isinstance(ctx.method_args[1], torch.Tensor):
        __convert_min_elementwise(ctx)
    else:
        __convert_min_reduce(ctx)
        

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_min_reduce_dim1():
    return UnaryModule(lambda x: torch.min(x, 1)[0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_min_reduce_dim22():
    return UnaryModule(lambda x: torch.min(x, 2)[0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_min_reduce_dim1_keepdim():
    return UnaryModule(lambda x: torch.min(x, 1, keepdim=True)[0])


class MinElementwise(torch.nn.Module):
    def forward(self, x, y):
        return torch.min(x, y)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1,)]) # broadcast
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3), (1, 3, 3)]) # broadcast
def test_min_elementwise():
    return MinElementwise()