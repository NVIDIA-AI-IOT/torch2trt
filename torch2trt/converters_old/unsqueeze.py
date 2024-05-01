import tensorrt as trt
import numpy as np
import torch
from torch2trt.torch2trt import tensorrt_converter, get_arg, torch_dim_resolve_negative, add_missing_trt_tensors, torch_dim_to_trt_axes
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.Tensor.unsqueeze')
@tensorrt_converter('torch.unsqueeze')
def convert_unsqueeze(ctx):
    input = ctx.method_args[0]

    if not hasattr(input, '_trt'):
        return

    dim = get_arg(ctx, 'dim', pos=1, default=None)
    assert(dim is not None)
    output = ctx.method_return

    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    input_shape_trt = ctx.network.add_shape(input_trt).get_output(0)
    new_shape_trt = []

    for i in range(input.ndim):
        # copy input dim
        new_shape_trt.append(
            ctx.network.add_slice(input_shape_trt, [i], [1], [1]).get_output(0)
        )
    
    # add unsqueeze dim
    new_shape_trt.insert(
        dim,
        ctx.network.add_constant([1], np.array([1], dtype=np.int32)).get_output(0)
    )

    new_shape_trt = ctx.network.add_concatenation(new_shape_trt).get_output(0)

    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, new_shape_trt)
    output._trt = layer.get_output(0)



class UnSqueeze(torch.nn.Module):
    def __init__(self, dim):
        super(UnSqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(dim=self.dim)        



@add_module_test(torch.float32, torch.device('cuda'), [(1, 7)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 5, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 1, 3)], max_batch_size=2)
def test_unsqueeze():
    return UnSqueeze(2)
