from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.layer_norm')
def convert_layernorm(ctx):
    input = get_arg(ctx, 'input', 0, None)
    shape = get_arg(ctx, 'normalized_shape', 1, None)
    weight = get_arg(ctx, 'weight', 2, None)
    bias = get_arg(ctx, 'bias', 3, None)
    eps = get_arg(ctx, 'eps', 4, 1e-05)
    output = ctx.method_return
    
    input_trt, eps_trt = add_missing_trt_tensors(
        ctx.network,
        [input, eps]
    )
    
    input_trt, eps_trt = broadcast_trt_tensors(
        ctx.network, 
        [input_trt, eps_trt],
        len(output.shape)
    )
    
    if weight is not None:
        _, weight_trt = add_missing_trt_tensors(
            ctx.network,
            [input, weight]
        )
        _, weight_trt = broadcast_trt_tensors(
            ctx.network, 
            [input_trt, weight_trt],
            len(output.shape)
        )
    
    if bias is not None:
        _, bias_trt = add_missing_trt_tensors(
            ctx.network,
            [input, bias]
        )
        _, bias_trt = broadcast_trt_tensors(
            ctx.network, 
            [input_trt, bias_trt],
            len(output.shape)
        )
    
    if isinstance(shape, int):
        shape = (shape,)
    dim = tuple([-i - 1 for i in range(len(shape))])
    dim = torch_dim_resolve_negative(dim, len(input.shape))
    axes = torch_dim_to_trt_axes(dim)
    
    ux = ctx.network.add_reduce(input_trt, trt.ReduceOperation.AVG, axes, keep_dims=True).get_output(0)
    numerator = ctx.network.add_elementwise(input_trt, ux, trt.ElementWiseOperation.SUB).get_output(0)
    varx = ctx.network.add_elementwise(numerator, numerator, trt.ElementWiseOperation.PROD).get_output(0)
    varx = ctx.network.add_reduce(varx, trt.ReduceOperation.AVG, axes, keep_dims=True).get_output(0)
    denom = ctx.network.add_elementwise(varx, eps_trt, trt.ElementWiseOperation.SUM).get_output(0)
    denom = ctx.network.add_unary(denom, trt.UnaryOperation.SQRT).get_output(0)
    y = ctx.network.add_elementwise(numerator, denom, trt.ElementWiseOperation.DIV).get_output(0)
    
    if weight is not None:
        y = ctx.network.add_elementwise(y, weight_trt, trt.ElementWiseOperation.PROD).get_output(0)
        
    if bias is not None:
        y = ctx.network.add_elementwise(y, bias_trt, trt.ElementWiseOperation.SUM).get_output(0)
    
    output._trt = y
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 5, 3)])
def test_layer_norm_1d():
    return torch.nn.LayerNorm(3)

    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 5, 3)])
def test_layer_norm_2d():
    return torch.nn.LayerNorm((5, 3))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 5, 3)])
def test_layer_norm_3d():
    return torch.nn.LayerNorm((5, 5, 3))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 5, 3)])
def test_layer_norm_1d_nonaffine():
    return torch.nn.LayerNorm(3, elementwise_affine=False)

    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 5, 3)])
def test_layer_norm_2d_nonaffine():
    return torch.nn.LayerNorm((5, 3), elementwise_affine=False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 5, 3)])
def test_layer_norm_3d_nonaffine():
    return torch.nn.LayerNorm((5, 5, 3), elementwise_affine=False)