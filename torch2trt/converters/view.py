from torch2trt.torch2trt import *
# from torch2trt.shape_conversion import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.Tensor.view')
@tensorrt_converter('torch.Tensor.reshape')
def convert_view(ctx):
    input = ctx.method_args[0]
    if not hasattr(input, '_trt'):
        return
    
    try:
        iter(ctx.method_args[1])
        size = make_size_wrapper(ctx.method_args[1])
    except:
        size = make_size_wrapper(ctx.method_args[1:])

    output = ctx.method_return

    layer = ctx.network.add_shuffle(input._trt)
    layer.set_input(1, size._trt)
    output._trt = layer.get_output(0)


class View(torch.nn.Module):
    def __init__(self, *dims):
        super(View, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.view(*self.dims)  


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], max_batch_size=2)
def test_view_1d():
    return View(1, -1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)], max_batch_size=2)
def test_view_2d():
    return View(1, 1, -1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3, 6)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3, 3, 6)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3, 3, 6)], max_batch_size=2)
def test_view_3d():
    return View(1, 3, 3, -1)


