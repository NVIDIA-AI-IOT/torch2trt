import torch
import torch.nn as nn
from torch2trt import torch2trt


def test_flatten_nested_tuple_args():

    class TestModule(nn.Module):

        def forward(self, x, yz):
            return torch.cat([x, yz[0], yz[1]], dim=-1)

    module = TestModule().cuda().eval()

    data = (
        torch.randn(1, 3, 32, 32).cuda(),
        (
            torch.randn(1, 3, 32, 32).cuda(),
            torch.randn(1, 3, 32, 32).cuda()
        )
    )

    module_trt = torch2trt(module, data)

    out = module(*data)
    out_trt = module_trt(*data)

    assert(torch.allclose(out, out_trt, atol=1e-3, rtol=1e-3))

