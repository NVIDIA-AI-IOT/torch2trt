from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.flatten')
@tensorrt_converter('torch.Tensor.reshape')
@tensorrt_converter('torch.Tensor.view')
@tensorrt_converter('torch.Tensor.squeeze')
@tensorrt_converter('torch.Tensor.unsqueeze')
@tensorrt_converter('torch.squeeze')
@tensorrt_converter('torch.unsqueeze')
def convert_view(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims = tuple(output.shape[1:])
    output._trt = layer.get_output(0)


class View(torch.nn.Module):
    def __init__(self, *dims):
        super(View, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.view(*self.dims)


class Squeeze(torch.nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)

class UnSqueeze(torch.nn.Module):
    def __init__(self, dim):
        super(UnSqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(dim=self.dim)        


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_view_1d():
    return View(1, -1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_view_2d():
    return View(1, 1, -1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3, 6)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3, 3, 6)])
def test_view_3d():
    return View(1, 3, 3, -1)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 7)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 5, 3)])
def test_unsqueeze():
    return UnSqueeze(2)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 1)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 1, 3)])
def test_squeeze():
    return Squeeze(2)    


