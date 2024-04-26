from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
import numpy as np


@tensorrt_converter('torch.nn.functional.linear')
def convert_Linear(ctx):
    input = ctx.method_args[0]
    weight = get_arg(ctx, 'weight', 1, None)
    bias = get_arg(ctx, 'bias', 2, None)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    weight = weight.detach().cpu().numpy()



    if bias is not None:
        bias = bias.detach().cpu().numpy()
    else:
        bias = np.zeros((int(weight.shape[0]),), dtype=weight.dtype)

    bias_shape = [1] * (input.ndim - 1) + [int(weight.shape[0])]
    bias = bias.reshape(bias_shape)

    kernel_const = ctx.network.add_constant(tuple(weight.shape), weight)
    bias_const = ctx.network.add_constant(tuple(bias.shape), bias)
    

    mm = ctx.network.add_matrix_multiply(
        input_trt,
        trt.MatrixOperation.NONE,
        kernel_const.get_output(0),
        trt.MatrixOperation.TRANSPOSE
    )

    bias_add = ctx.network.add_elementwise(
        mm.get_output(0), 
        bias_const.get_output(0), 
        trt.ElementWiseOperation.SUM
    
    )

    output._trt = bias_add.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
def test_Linear_basic():
    return torch.nn.Linear(10, 5)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(2, 3, 4, 10)], max_batch_size=2)
def test_Linear_no_bias():
    return torch.nn.Linear(10, 5, bias=False)
