import torch.nn as nn
from torch2trt.torch2trt import *                                 
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.group_norm')
def convert_group_norm(ctx):

    input = get_arg(ctx, 'input', pos=0, default=None)
    num_groups = get_arg(ctx, 'num_groups', pos=1, default=None)
    weight = get_arg(ctx, 'weight', pos=2, default=None)
    bias = get_arg(ctx, 'bias', pos=3, default=None)
    eps = get_arg(ctx, 'eps', pos=4, default=1e-5)
    output = ctx.method_return


    input_trt, eps_trt = add_missing_trt_tensors(ctx.network, [input, eps])
    
    shape = list(input.shape)
    split_shape = [shape[0]] + [num_groups, shape[1] // num_groups] + shape[2:]
    split_shape = tuple(split_shape)
    keepdim = True

    # split into groups
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims = split_shape
    a = layer.get_output(0)


    # compute mean over groups
    reduce_dims = tuple(range(2, len(split_shape)))
    axes = torch_dim_to_trt_axes(reduce_dims)
    layer = ctx.network.add_reduce(a, trt.ReduceOperation.AVG, axes, keepdim)
    a_mean = layer.get_output(0)

    # compute stdev over groups
    a_diff = ctx.network.add_elementwise(a, a_mean, trt.ElementWiseOperation.SUB).get_output(0)
    a_dist = ctx.network.add_elementwise(a_diff, a_diff, trt.ElementWiseOperation.PROD).get_output(0)
    a_var = ctx.network.add_reduce(a_dist, trt.ReduceOperation.AVG, axes, keepdim).get_output(0)


    a_var, eps_trt = broadcast_trt_tensors(ctx.network, [a_var, eps_trt], len(split_shape))

    a_var_eps = ctx.network.add_elementwise(a_var, eps_trt, trt.ElementWiseOperation.SUM).get_output(0)
    a_std = ctx.network.add_unary(a_var_eps, trt.UnaryOperation.SQRT).get_output(0)

    # divide by stdev
    b = ctx.network.add_elementwise(a_diff, a_std, trt.ElementWiseOperation.DIV).get_output(0)

    # reshape
    layer = ctx.network.add_shuffle(b)
    layer.reshape_dims = shape

    c = layer.get_output(0)

    # handle affine version
    if weight is not None or bias is not None:
        if weight is not None:
            scale = weight.detach().cpu().numpy()
        else:
            scale = np.ones(input.shape[1])

        if bias is not None:
            bias = bias.detach().cpu().numpy()
        else:
            bias = np.zeros(input.shape[1])

        power = np.ones_like(scale)

        layer = ctx.network.add_scale_nd(c, trt.ScaleMode.CHANNEL, bias, scale, power, 1)
        c = layer.get_output(0)

    output._trt = c


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)])
def test_group_norm_trt_g2_fp32():
    return torch.nn.GroupNorm(2, 10)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)])
def test_group_norm_trt_g2_eps_fp32():
    return torch.nn.GroupNorm(2, 10, eps=1e-4)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)])
def test_group_norm_trt_g2_eps_fp32_affine():
    module = torch.nn.GroupNorm(2, 10, affine=True, eps=1e-4)
    module.weight.data = torch.randn_like(module.weight.data)
    module.bias.data = torch.randn_like(module.bias.data)
    return module


    
