from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


def __get_arg(ctx, name, pos, default):
    if name in ctx.method_kwargs:
        return ctx.method_kwargs[name]
    elif len(ctx.method_args) > pos:
        return ctx.method_args[pos]
    else:
        return default
    

def __trt_add_scalar_constant_like(network, tensor, value):
    shape = (1, ) * len(tensor.shape)  # broadcast all dimensions
    array = value * torch.ones(shape, dtype=torch_dtype_from_trt(tensor.dtype)).cpu().numpy()
    return network.add_constant(shape, array).get_output(0)


def __torch_dim_to_trt_bitmask(dim):
    if not isinstance(dim, tuple):
        dim = (dim, )
        
    # create axes bitmask for reduce layer
    axes = 0
    for d in dim:
        axes |= 1 << (d - 1) # -1 to remove batch dimension
        
    return axes


@tensorrt_converter('torch.nn.functional.normalize')
def convert_normalize(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return
    
    # get power
    p = __get_arg(ctx, name='p', pos=1, default=2)
    dim = __get_arg(ctx, name='dim', pos=2, default=1)
    eps = __get_arg(ctx, name='eps', pos=3, default=1e-12)
    
    eps_trt = __trt_add_scalar_constant_like(ctx.network, input._trt, eps)
    p_trt = __trt_add_scalar_constant_like(ctx.network, input._trt, p)
    p_inv_trt = __trt_add_scalar_constant_like(ctx.network, input._trt, 1.0 / p)
    
    # compute norm = sum(abs(x)**p, dim=dim)**(1./p)
    norm = ctx.network.add_unary(input._trt, trt.UnaryOperation.ABS).get_output(0)
    norm = ctx.network.add_elementwise(norm, p_trt, trt.ElementWiseOperation.POW).get_output(0)
    norm = ctx.network.add_reduce(norm, trt.ReduceOperation.SUM, __torch_dim_to_trt_bitmask(dim), keep_dims=True).get_output(0)
    norm = ctx.network.add_elementwise(norm, p_inv_trt, trt.ElementWiseOperation.POW).get_output(0)
    
    # clamp norm = max(norm, eps)
    norm = ctx.network.add_elementwise(norm, eps_trt, trt.ElementWiseOperation.MAX).get_output(0)
    
    # divide input by norm
    output._trt = ctx.network.add_elementwise(input._trt, norm, trt.ElementWiseOperation.DIV).get_output(0)
    

class Normalize(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Normalize, self).__init__()
        self.args = args
        self.kwargs = kwargs
    
    def forward(self, x):
        return torch.nn.functional.normalize(x, *self.args, **self.kwargs)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_normalize_basic():
    return Normalize()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_normalize_l1_basic():
    return Normalize(p=1.0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_normalize_l1p5_basic():
    return Normalize(p=1.5)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_normalize_l2_height():
    return Normalize(p=2.0, dim=2)