from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

    
@tensorrt_converter('torch.roll')
@tensorrt_converter('torch.Tensor.roll')
def convert_roll(ctx):
    input = get_arg(ctx, 'input', 0, None)
    shifts = get_arg(ctx, 'shifts', 1, None)
    dims = get_arg(ctx, 'dims', 2, None)
    output = ctx.method_return
    
    assert dims is not None, "roll converter only supports roll when dims is specified"
    
    ndim = input.ndim
    
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    try:
        iter(shifts)
    except:
        shifts = (shifts,)
        dims = (dims,)
    
    start = [0] * ndim
    shape = tuple([int(d) for d in input.shape])
    stride = [1] * ndim
    
    for s, d in zip(shifts, dims):
        start[d] = (-s) % shape[d]
    
    start = tuple(start)
    shape = tuple(shape)
    stride = tuple(stride)
    
    shape_dynamic = ctx.network.add_shape(input._trt).get_output(0)
    layer = ctx.network.add_slice(
        input_trt,
        start,  # [1:] to exclude batch
        shape,
        stride
    )
    layer.set_input(2, shape_dynamic)
    layer.mode = trt.SliceMode.WRAP
    
    output._trt = layer.get_output(0)
    
    
class Roll(torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        
    def forward(self, x):
        return torch.roll(x, *self.args, **self.kwargs)

    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 4)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 5)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
@add_module_test(torch.float32, torch.device('cuda'), [(2, 3, 4, 5)], max_batch_size=2)
def test_roll_int():
    return Roll(1, 1)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 5)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
@add_module_test(torch.float32, torch.device('cuda'), [(2, 3, 4, 5)], max_batch_size=2)
def test_roll_int_dim():
    return Roll(1, -2)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
@add_module_test(torch.float32, torch.device('cuda'), [(2, 3, 4, 5)], max_batch_size=2)
def test_roll_tuple():
    return Roll((2, 3), (1, 3))