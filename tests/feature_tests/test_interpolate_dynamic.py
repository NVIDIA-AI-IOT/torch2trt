import pytest
import torch
import torch.nn.functional as F
from torch2trt import (
    torch2trt,
    trt
)


def test_interpolate_dynamic_size():

    class TestModule(torch.nn.Module):
        def forward(self, x):
            size = x.size()
            return F.interpolate(x, size=(size[2]*2, size[3]*3))

    module = TestModule().cuda().eval()

    x = torch.randn(1, 3, 32, 32).cuda()

    module_trt = torch2trt(module, [x], log_level=trt.Logger.VERBOSE, min_shapes=[(1, 3, 32, 32)], max_shapes=[(4, 3, 64, 64)], opt_shapes=[(1, 3, 32, 32)])

    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))

    x = torch.randn(1, 3, 32, 32).cuda()
    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))
    
    x = torch.randn(4, 3, 64, 64).cuda()
    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))


def test_interpolate_dynamic_shape():

    class TestModule(torch.nn.Module):
        def forward(self, x):
            size = x.shape
            return F.interpolate(x, size=(size[2]*2, size[3]*3))

    module = TestModule().cuda().eval()

    x = torch.randn(1, 3, 32, 32).cuda()

    module_trt = torch2trt(module, [x], log_level=trt.Logger.VERBOSE, min_shapes=[(1, 3, 32, 32)], max_shapes=[(4, 3, 64, 64)], opt_shapes=[(1, 3, 32, 32)])

    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))

    x = torch.randn(1, 3, 32, 32).cuda()
    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))
    
    x = torch.randn(4, 3, 64, 64).cuda()
    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))
