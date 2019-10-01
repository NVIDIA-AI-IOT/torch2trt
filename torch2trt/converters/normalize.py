from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.normalize')
def convert_normalize(ctx):
    # get args
    input = get_arg(ctx, 'input', pos=0, default=None)
    p = get_arg(ctx, 'p', pos=1, default=2)
    dim = get_arg(ctx, 'dim', pos=2, default=1)
    eps = get_arg(ctx, 'eps', pos=3, default=1e-12)
    
#     input_trt = input._trt
    output = ctx.method_return
    
    # add broadcastable scalar constants to network
    input_trt, eps_trt, p_trt, p_inv_trt = trt_(ctx.network, input, eps, p, 1.0 / p)
    
    # compute norm = sum(abs(x)**p, dim=dim)**(1./p)
    norm = ctx.network.add_unary(input_trt, trt.UnaryOperation.ABS).get_output(0)
    norm = ctx.network.add_elementwise(norm, p_trt, trt.ElementWiseOperation.POW).get_output(0)
    norm = ctx.network.add_reduce(norm, trt.ReduceOperation.SUM, torch_dim_to_trt_axes(dim), keep_dims=True).get_output(0)
    norm = ctx.network.add_elementwise(norm, p_inv_trt, trt.ElementWiseOperation.POW).get_output(0)
    
    # clamp norm = max(norm, eps)
    norm = ctx.network.add_elementwise(norm, eps_trt, trt.ElementWiseOperation.MAX).get_output(0)
    
    # divide input by norm
    output._trt = ctx.network.add_elementwise(input_trt, norm, trt.ElementWiseOperation.DIV).get_output(0)
    

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