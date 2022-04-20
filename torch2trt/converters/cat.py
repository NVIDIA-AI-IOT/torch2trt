from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.cat')
def convert_cat(ctx):
    inputs = get_arg(ctx, 'input', pos=0, default=None)
    dim = get_arg(ctx, 'dim', pos=1, default=0)

    # Reverse negative dims.
    if dim < 0:
        dim = len(inputs[0].shape) - abs(dim)

    output = ctx.method_return
    trt_inputs = add_missing_trt_tensors(ctx.network, inputs)
    trt_inputs = broadcast_trt_tensors(ctx.network, trt_inputs, len(output.shape))

    layer = ctx.network.add_concatenation(inputs=trt_inputs)
    layer.axis = dim
    output._trt = layer.get_output(0)


class Cat(torch.nn.Module):
    def __init__(self, dim):
        super(Cat, self).__init__()
        self.dim = dim

    def forward(self, *x):
        return torch.cat(x, dim=self.dim)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 3, 4), (1, 17, 4)])
@add_module_test(torch.float32, torch.device('cuda'), [(2, 4, 4), (2, 3, 4), (2, 17, 4)], max_batch_size=2)
def test_Cat_basic():
    return Cat(1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 4, 4), (1, 4, 4)])
@add_module_test(torch.float32, torch.device('cuda'), [(2, 4, 4), (2, 4, 4), (2, 4, 4)], max_batch_size=2)
def test_Cat_neg1_dim():
    return Cat(-1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 4, 4), (1, 4, 4)])
@add_module_test(torch.float32, torch.device('cuda'), [(2, 4, 4), (2, 4, 4), (2, 4, 4)], max_batch_size=2)
def test_Cat_neg2_dim():
    return Cat(-2)
