from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from .unary import UnaryModule


def __convert_max_elementwise(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape) - 1)
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.MAX)
    output._trt = layer.get_output(0)
    

def __convert_max_reduce(ctx):
    input = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=tuple(range(1, len(input.shape))))
    keepdim = get_arg(ctx, 'keepdim', pos=2, default=False)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output_val = ctx.method_return[0]
    output_idx = ctx.method_return[1]
    layer = ctx.network.add_reduce(input_trt,  trt.ReduceOperation.MAX, torch_dim_to_trt_axes(dim), keepdim)
    output_val._trt = layer.get_output(0)
    

@tensorrt_converter('torch.max')
@tensorrt_converter('torch.Tensor.max')
def convert_max(ctx):
    if len(ctx.method_args) > 1 and isinstance(ctx.method_args[1], torch.Tensor):
        __convert_max_elementwise(ctx)
    else:
        __convert_max_reduce(ctx)
        

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_max_reduce_dim1():
    return UnaryModule(lambda x: torch.max(x, 1)[0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_max_reduce_dim22():
    return UnaryModule(lambda x: torch.max(x, 2)[0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_max_reduce_dim1_keepdim():
    return UnaryModule(lambda x: torch.max(x, 1, keepdim=True)[0])


class MaxElementwise(torch.nn.Module):
    def forward(self, x, y):
        return torch.max(x, y)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1,)]) # broadcast
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3), (1, 3, 3)]) # broadcast
def test_max_elementwise():
    return MaxElementwise()
