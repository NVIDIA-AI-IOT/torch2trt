import tensorrt as trt
import numpy as np
import torch
from torch2trt.torch2trt import tensorrt_converter, get_arg, torch_dim_resolve_negative, add_missing_trt_tensors, torch_dim_to_trt_axes
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.Tensor.squeeze')
@tensorrt_converter('torch.squeeze')
def convert_squeeze(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return
    dim = get_arg(ctx, 'dim', pos=1, default=None)

    if dim < 0:
        dim = len(input.shape) + dim
    assert dim >= 0

    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    input_shape_trt = ctx.network.add_shape(input_trt).get_output(0)
    new_shape_trt = []

    # get shape before flatten
    for i in range(input.ndim):
        if input.size(i) == 1 and (dim is None) or (i == dim):
            continue # skip 1 dimensions
        else:
            new_shape_trt.append(
                ctx.network.add_slice(input_shape_trt, [i], [1], [1]).get_output(0)
            )

    new_shape_trt = ctx.network.add_concatenation(new_shape_trt).get_output(0)

    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, new_shape_trt)
    output._trt = layer.get_output(0)


class Squeeze(torch.nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 1)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 1, 3)], max_batch_size=2)
def test_squeeze():
    return Squeeze(2)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 1, 1)])
def test_squeeze_neg():
    return Squeeze(-1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 1, 1)])
def test_squeeze_neg2():
    return Squeeze(-2)
