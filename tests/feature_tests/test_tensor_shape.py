import pytest
import torch
import torch.nn.functional as F
from torch2trt import (
    torch2trt,
    trt,
    SizeWrapper,
    tensorrt_converter
)


def test_tensor_shape_view_trivial():

    class TestModule(torch.nn.Module):
        def forward(self, x):
            size = x.size()
            return x.view(size)

    module = TestModule().cuda().eval()

    x = torch.randn(1, 3, 32, 32).cuda()

    module_trt = torch2trt(module, [x], log_level=trt.Logger.VERBOSE, max_batch_size=4)

    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))

    x = torch.randn(1, 3, 32, 32).cuda()
    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))
    
    x = torch.randn(4, 3, 32, 32).cuda()
    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))


def test_tensor_shape_view_mul():

    class TestModule(torch.nn.Module):
        def forward(self, x):
            size = x.size()
            return x.view(size[0] * size[1], size[2] * size[3])

    module = TestModule().cuda().eval()

    x = torch.randn(1, 3, 32, 32).cuda()

    module_trt = torch2trt(module, [x], log_level=trt.Logger.VERBOSE, max_batch_size=4)

    x = torch.randn(1, 3, 32, 32).cuda()
    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))

    x = torch.randn(4, 3, 32, 32).cuda()
    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))


def test_tensor_shape_view_mul():

    class TestModule(torch.nn.Module):
        def forward(self, x):
            size = x.size()
            return x.view(size[0] * size[1], size[2] * size[3])

    module = TestModule().cuda().eval()

    x = torch.randn(1, 3, 32, 32).cuda()

    module_trt = torch2trt(module, [x], log_level=trt.Logger.VERBOSE, max_batch_size=4)

    x = torch.randn(1, 3, 32, 32).cuda()
    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))

    x = torch.randn(4, 3, 32, 32).cuda()
    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))


def test_tensor_shape_view_mul_cast():

    class TestModule(torch.nn.Module):
        def forward(self, x):
            size = x.size()
            return x.view(size[0] * int(size[1]), int(size[2] * size[3]))

    module = TestModule().cuda().eval()

    x = torch.randn(1, 3, 32, 32).cuda()

    module_trt = torch2trt(module, [x], log_level=trt.Logger.VERBOSE, max_batch_size=4)

    x = torch.randn(1, 3, 32, 32).cuda()
    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))

    x = torch.randn(4, 3, 32, 32).cuda()
    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))


def test_tensor_shape_view_mul_const_lhs():

    class TestModule(torch.nn.Module):
        def forward(self, x):
            size = x.size()
            return x.view(size[0] * 1, size[1], size[2] * size[3])

    module = TestModule().cuda().eval()

    x = torch.randn(1, 3, 32, 32).cuda()

    module_trt = torch2trt(module, [x], log_level=trt.Logger.VERBOSE, max_batch_size=4)

    x = torch.randn(1, 3, 32, 32).cuda()
    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))

    x = torch.randn(4, 3, 32, 32).cuda()
    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))


def test_tensor_shape_view_mul_const_rhs():

    class TestModule(torch.nn.Module):
        def forward(self, x):
            size = x.size()
            return x.view(1 * size[0], size[1], size[2] * size[3])

    module = TestModule().cuda().eval()

    x = torch.randn(1, 3, 32, 32).cuda()

    module_trt = torch2trt(module, [x], log_level=trt.Logger.VERBOSE, max_batch_size=4)

    x = torch.randn(1, 3, 32, 32).cuda()
    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))

    x = torch.randn(4, 3, 32, 32).cuda()
    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))


def test_tensor_shape_view_static():

    class TestModule(torch.nn.Module):
        def forward(self, x):
            size = x.size()
            return x.view(1, 3, 32, 32)

    module = TestModule().cuda().eval()

    x = torch.randn(1, 3, 32, 32).cuda()

    module_trt = torch2trt(module, [x], log_level=trt.Logger.VERBOSE, max_batch_size=4)

    x = torch.randn(1, 3, 32, 32).cuda()
    assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))

    # x = torch.randn(4, 3, 32, 32).cuda()
    # assert(torch.allclose(module_trt(x), module(x), atol=1e-2, rtol=1e-2))


if __name__ == '__main__':

    test_tensor_shape_view_mul()