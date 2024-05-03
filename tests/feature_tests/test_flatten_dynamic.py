import pytest
from torch2trt import torch2trt, trt
import torch


class FlattenModule(torch.nn.Module):
    def __init__(self, start_dim, end_dim):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return torch.flatten(x, self.start_dim, self.end_dim)


def test_flatten_dynamic_0_n1():

    # 0, -1
    module = FlattenModule(start_dim=0, end_dim=-1).cuda().eval()

    x = torch.randn(1, 4, 5).cuda()

    module_trt = torch2trt(module, [x], max_batch_size=4, log_level=trt.Logger.VERBOSE)

    x = torch.randn(1, 4, 5).cuda()
    assert(torch.allclose(module(x), module_trt(x), atol=1e-2, rtol=1e-2))

    x = torch.randn(4, 4, 5).cuda()
    assert(torch.allclose(module(x), module_trt(x), atol=1e-2, rtol=1e-2))


def test_flatten_dynamic_1_n1():
    # 1, -1
    module = FlattenModule(start_dim=1, end_dim=-1).cuda().eval()

    x = torch.randn(1, 4, 5).cuda()

    module_trt = torch2trt(module, [x], max_batch_size=4, log_level=trt.Logger.VERBOSE)

    x = torch.randn(1, 4, 5).cuda()
    assert(torch.allclose(module(x), module_trt(x), atol=1e-2, rtol=1e-2))

    x = torch.randn(4, 4, 5).cuda()
    assert(torch.allclose(module(x), module_trt(x), atol=1e-2, rtol=1e-2))


def test_flatten_dynamic_0_1():
    # 0, 1
    module = FlattenModule(start_dim=0, end_dim=1).cuda().eval()

    x = torch.randn(1, 4, 5).cuda()

    module_trt = torch2trt(module, [x], max_batch_size=4, log_level=trt.Logger.VERBOSE)

    x = torch.randn(1, 4, 5).cuda()
    assert(torch.allclose(module(x), module_trt(x), atol=1e-2, rtol=1e-2))

    x = torch.randn(4, 4, 5).cuda()
    assert(torch.allclose(module(x), module_trt(x), atol=1e-2, rtol=1e-2))
    

if __name__ == '__main__':

    test_flatten_dynamic_0_1()