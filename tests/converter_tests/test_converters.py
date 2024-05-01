import pytest
import torch
import torch2trt
import torch.nn as nn
from torch2trt.flattener import Flattener


def cross_validate(
        module, 
        inputs,
        fp16_mode: bool,
        tol: float
    ):

    module = module
    

    module_trt = torch2trt.torch2trt(
        module,
        inputs,
        fp16_mode=fp16_mode
    )
    

    output = module(*inputs)
    output_trt = module_trt(*inputs)

    assert torch.allclose(output, output_trt, atol=tol, rtol=tol)



# MODULES
    

class UnaryModule(torch.nn.Module):
    def __init__(self, fn):
        super(UnaryModule, self).__init__()
        self.fn = fn
        
    def forward(self, x):
        return self.fn(x)


class BinaryModule(torch.nn.Module):
    def __init__(self, fn):
        super(BinaryModule, self).__init__()
        self.fn = fn
        
    def forward(self, a, b):
        return self.fn(a, b)
# TESTS



@pytest.mark.parametrize("fp16_mode,tol", [(False, 1e-1), (True, 1e-1)])
def test_leaky_relu(fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.leaky_relu(x)).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


@pytest.mark.parametrize("fp16_mode,tol", [(False, 1e-1), (True, 1e-1)])
def test_elu(fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.elu(x)).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


@pytest.mark.parametrize("fp16_mode,tol", [(False, 1e-1), (True, 1e-1)])
def test_selu(fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.selu(x)).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


@pytest.mark.parametrize("fp16_mode,tol", [(False, 1e-1), (True, 1e-1)])
def test_softsign(fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.selu(x)).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


@pytest.mark.parametrize("fp16_mode,tol", [(False, 1e-1), (True, 1e-1)])
def test_softplus(fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.softplus(x)).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


@pytest.mark.parametrize("output_size,fp16_mode,tol", [
    ((1, 1), False, 1e-1), 
    ((2, 2), False, 1e-1),
    ((1, 1), True, 1e-1)
])
def test_adaptive_avg_pool2d(output_size, fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.adaptive_avg_pool2d(x, output_size)).cuda().eval()
    inputs = [torch.randn(1, 3, 4, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


@pytest.mark.parametrize("output_size,fp16_mode,tol", [
    ((1, 1, 1), False, 1e-1), 
    ((2, 2, 2), False, 1e-1), 
    ((1, 1, 1), True, 1e-1)
])
def test_adaptive_avg_pool3d(output_size, fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.adaptive_avg_pool3d(x, output_size)).cuda().eval()
    inputs = [torch.randn(1, 3, 4, 4, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


@pytest.mark.parametrize("output_size,fp16_mode,tol", [
    ((1, 1), False, 1e-1), 
    ((2, 2), False, 1e-1),
    ((1, 1), True, 1e-1)
])
def test_adaptive_max_pool2d(output_size, fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.adaptive_max_pool2d(x, output_size)).cuda().eval()
    inputs = [torch.randn(1, 3, 4, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


@pytest.mark.parametrize("output_size,fp16_mode,tol", [
    ((1, 1, 1), False, 1e-1), 
    ((2, 2, 2), False, 1e-1), 
    ((1, 1, 1), True, 1e-1)
])
def test_adaptive_max_pool3d(output_size, fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.adaptive_max_pool3d(x, output_size)).cuda().eval()
    inputs = [torch.randn(1, 3, 4, 4, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


def test_add():
    module = BinaryModule(lambda a, b: a + b).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda(), torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-2)


def test_torch_add():
    module = BinaryModule(lambda a, b: torch.add(a, b)).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda(), torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-2)


def test_iadd():
    class IAdd(torch.nn.Module):
        def __init__(self):
            super(IAdd, self).__init__()

        def forward(self, x, y):
            x += y
            return x

    module = IAdd().cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda(), torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-2)


def test_radd_int():
    module = UnaryModule(lambda x: 1 + x).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-2)


def test_radd_float():
    module = UnaryModule(lambda x: 1.0 + x).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-2)


# TODO: radd, add, iadd
    

@pytest.mark.parametrize("kernel_size,stride,padding,ceil_mode,count_include_pad", [
    (3, 2, 1, False, True),
    (3, 2, 1, True, False)
])
def test_avg_pool2d(kernel_size, stride, padding, ceil_mode, count_include_pad):
    module = UnaryModule(lambda x: torch.nn.functional.avg_pool2d(
        x, kernel_size, stride, padding, ceil_mode, count_include_pad
    )).cuda().eval()
    inputs = [torch.randn(1, 3, 8, 8).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize("kernel_size,stride,padding,ceil_mode,count_include_pad", [
    (3, 2, 1, False, True),
    (3, 2, 1, True, False)
])
def test_avg_pool3d(kernel_size, stride, padding, ceil_mode, count_include_pad):
    module = UnaryModule(lambda x: torch.nn.functional.avg_pool3d(
        x, kernel_size, stride, padding, ceil_mode, count_include_pad
    )).cuda().eval()
    inputs = [torch.randn(1, 3, 8, 8, 8).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


def test_batch_norm_1d():
    module = nn.BatchNorm2d(3).cuda().eval()
    inputs = [torch.randn(2, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


def test_batch_norm_2d():
    module = nn.BatchNorm2d(3).cuda().eval()
    inputs = [torch.randn(2, 3, 4, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


def test_batch_norm_3d():
    module = nn.BatchNorm2d(3).cuda().eval()
    inputs = [torch.randn(2, 3, 4, 4, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)

