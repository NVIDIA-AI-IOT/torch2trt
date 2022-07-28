import pytest

import torch
import torch.nn.functional as F
from torch2trt import torch2trt, tensorrt_converter, get_arg, add_missing_trt_tensors


class _IntWrapper(int):
    pass


class _ShapeWrapper(object):
    
    def __init__(self, shape):
        self._shape = shape
    
    def __getattr__(self, name):
        return getattr(self._shape, name)

    def __getitem__(self, key):
        return self._shape.__getitem__(key)


_original_getattribute = torch.Tensor.__getattribute__


def _modified_getattribute(self, name):
    if name == 'shape':
        return self.size()
    else:
        return _original_getattribute(self, name)



@tensorrt_converter('torch.Tensor.size')
def convert_size(ctx):
    input = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=None)

    add_missing_trt_tensors(ctx.network, [input])

    if dim is None:
        ctx.method_return = _ShapeWrapper(ctx.method_return)
        output = ctx.method_return
        output._trt = ctx.network.add_shape(input._trt).get_output(0)
    else:
        ctx.method_return = _IntWrapper(ctx.method_return)
        shape_trt = ctx.network.add_shape(input._trt)
        


    output_trt = ctx.network.add_shape(input._trt)
    
    if dim is not None:



    


def test_tensor_shape_view():

    class TestModule(torch.nn.Module):
        def forward(self, x):
            return x.view(x.size())

    module = TestModule().cuda().eval()

    x = torch.randn(1, 3, 32, 32)

    module_trt = torch2trt(module, [x])

    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))