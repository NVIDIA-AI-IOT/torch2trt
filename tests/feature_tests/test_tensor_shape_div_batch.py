import pytest
import torch
from torch2trt import torch2trt, trt
    
def test_div_constant_batch():

    class DivConstantBatch(torch.nn.Module):
        def __init__(self):
            super(DivConstantBatch, self).__init__()
            self.register_buffer('y', torch.ones((1, 3, 10, 10)))

        def forward(self, x):
            return x / self.y

    module = DivConstantBatch().cuda().eval()

    x = torch.randn(1, 3, 10, 10).cuda()

    module_trt = torch2trt(module, [x], log_level=trt.Logger.VERBOSE)

    assert torch.allclose(module_trt(x), module(x), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    test_div_constant_batch()
