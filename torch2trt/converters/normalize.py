from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.normalize')
def convert_normalize(ctx):
    input = ctx.method_args[0]
    input_trt = get_or_create_trt_tensor(ctx.network, input)
    output = ctx.method_return
    
    # get power
    p = ctx.parsed_args['p']
    dim = ctx.parsed_args['dim']
    eps = ctx.parsed_args['eps']
    
    # add broadcastable scalar constants to network
    scalar_shape = (1,) * len(input.shape)
    eps_trt = get_or_create_trt_tensor(ctx.network, eps * torch.ones(scalar_shape, dtype=input.dtype))
    p_trt = get_or_create_trt_tensor(ctx.network, p * torch.ones(scalar_shape, dtype=input.dtype))
    p_inv_trt = get_or_create_trt_tensor(ctx.network, torch.ones(scalar_shape, dtype=input.dtype) / p)
    
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