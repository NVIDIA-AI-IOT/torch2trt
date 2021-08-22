from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.Tensor.expand_as')
def convert_expand_as(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return
    
    inshape = tuple(input.shape)[1:] # exclude batch
    shape = tuple(output.shape)[1:]
    ndim = len(shape)
    start = tuple([0]*ndim)
    stride = tuple([int(i == o) for i, o in zip(inshape, shape)])  # stride == 1 if dimensions match, 0 otherwise
    
    layer = ctx.network.add_slice(input._trt, start, shape, stride)
    
    output._trt = layer.get_output(0)
    
    
class ExpandAsModule(torch.nn.Module):
    def __init__(self, other: torch.Tensor):
        super(ExpandAsModule, self).__init__()
        self.other = other
    
    def forward(self, x: torch.Tensor):
        return x.expand_as(self.other)


@add_module_test(torch.float32, torch.device('cuda'), [(1),])
def test_tensor_expand_as_scalar():
    return ExpandAsModule(torch.randn(3))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 1, 3, 3),])
def test_tensor_expand_as_singledim():
    return ExpandAsModule(torch.randn((1, 3, 3, 3)))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 1, 1, 3),])
def test_tensor_expand_as_multidim():
    return ExpandAsModule(torch.randn((1, 3, 3, 3)))


@add_module_test(torch.float16, torch.device('cuda'), [(1, 1, 3, 3),])
def test_tensor_expand_as_singledim_half():
    return ExpandAsModule(torch.randn((1, 3, 3, 3)))


@add_module_test(torch.float16, torch.device('cuda'), [(1, 1, 1, 3),])
def test_tensor_expand_as_multidim_half():
    return ExpandAsModule(torch.randn((1, 3, 3, 3)))
