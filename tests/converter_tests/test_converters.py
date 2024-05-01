import pytest
import torch
import torch2trt
from torch2trt.flattener import Flattener


def _cross_validate(
        module, 
        inputs,
        *args,
        **kwargs
    ):

    module = module
    

    module_trt = torch2trt.torch2trt(
        module,
        inputs,
        *args,
        **kwargs
    )
    

    output = module(*inputs)
    output_trt = module_trt(*inputs)

    assert torch.allclose(output, output_trt, atol=1e-2, rtol=1e-2)


class UnaryModule(torch.nn.Module):
    def __init__(self, fn):
        super(UnaryModule, self).__init__()
        self.fn = fn
        
    def forward(self, x):
        return self.fn(x)
    

def test_functional_leaky_relu():
    _cross_validate(
        UnaryModule(lambda x: torch.nn.functional.leaky_relu(x)).cuda().eval(),
        [torch.randn(1, 5, 3).cuda()]
    )


def test_functional_elu():
    _cross_validate(
        UnaryModule(lambda x: torch.nn.functional.elu(x)).cuda().eval(),
        [torch.randn(1, 5, 3).cuda()]
    )


def test_selu():
    _cross_validate(
        UnaryModule(lambda x: torch.selu(x)).cuda().eval(),
        [torch.randn(1, 5, 3).cuda()]
    )


def test_functional_selu():
    _cross_validate(
        UnaryModule(lambda x: torch.nn.functional.selu(x)).cuda().eval(),
        [torch.randn(1, 5, 3).cuda()]
    )


def test_functional_softsign():
    _cross_validate(
        UnaryModule(lambda x: torch.nn.functional.softsign(x)).cuda().eval(),
        [torch.randn(1, 5, 3).cuda()]
    )


def test_functional_softplus():
    _cross_validate(
        UnaryModule(lambda x: torch.nn.functional.softplus(x)).cuda().eval(),
        [torch.randn(1, 5, 3).cuda()]
    )
