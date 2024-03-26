import torch
import torch.nn as nn

from torch2trt import torch2trt


class FlipModule(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.flip(x, self.dims)


class FlipTensorModule(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.flip(self.dims)



def test_torch_flip():

    x = torch.randn(1, 2, 3).cuda()

    model = FlipModule(dims=(1,)).cuda().eval()
    model_trt = torch2trt(model, [x])

    out = model(x)
    out_trt = model_trt(x)

    assert torch.allclose(out, out_trt, rtol=1e-4, atol=1e-4)

def test_torch_flip_multidim():

    x = torch.randn(1, 2, 3).cuda()

    model = FlipTensorModule(dims=(1, 2)).cuda().eval()
    model_trt = torch2trt(model, [x])

    out = model(x)
    out_trt = model_trt(x)

    assert torch.allclose(out, out_trt, rtol=1e-4, atol=1e-4)

def test_torch_flip_tensor():

    x = torch.randn(1, 2, 3).cuda()

    model = FlipTensorModule(dims=(1,)).cuda().eval()
    model_trt = torch2trt(model, [x])

    out = model(x)
    out_trt = model_trt(x)

    assert torch.allclose(out, out_trt, rtol=1e-4, atol=1e-4)
