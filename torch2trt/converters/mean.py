from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.mean')
@tensorrt_converter('torch.Tensor.mean')
def convert_mean(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    
    # get dims from args or kwargs
    if 'dim' in ctx.method_kwargs: 
        dim = ctx.method_kwargs['dim']
    elif len(ctx.method_args) >= 2:
        dim = ctx.method_args[1]
        
    # convert list to tuple
    if isinstance(dim, list):
        dim = tuple(dim)
        
    if not isinstance(dim, tuple):
        dim = (dim, )
        
    # create axes bitmask for reduce layer
    axes = 0
    for d in dim:
        axes |= 1 << (d - 1) # -1 to remove batch dimension
        
    # get whether to keep dimensions
    if 'keepdim' in ctx.method_kwargs:
        keep_dims = ctx.method_kwargs['keepdim']
    elif len(ctx.method_args) == 3:
        keep_dims = ctx.method_args[2]
    else:
        keep_dims = False
        
    layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.AVG, axes, keep_dims)
    output._trt = layer.get_output(0)

    
class Mean(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super(Mean, self).__init__()
        self.dim = dim
        self.keepdim = keepdim
    def forward(self, x):
        return x.mean(self.dim, self.keepdim)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_channel():
    return Mean(1, False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_tuple():
    return Mean((1, 2), False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_keepdim():
    return Mean(1, True)