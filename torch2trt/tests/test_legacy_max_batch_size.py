import torch.nn as nn
import torch
from torch2trt import torch2trt


def test_legacy_max_batch_size():

    model = nn.Conv2d(3, 6, kernel_size=1).cuda().eval()

    data = torch.randn(1, 3, 32, 32).cuda()

    model_trt = torch2trt(model, [data], max_batch_size=4)


    data = torch.randn(1, 3, 32, 32).cuda()
    out = model(data)
    out_trt = model_trt(data)

    assert(torch.allclose(out, out_trt, atol=1e-3, rtol=1e-3))


    data = torch.randn(4, 3, 32, 32).cuda()
    out = model(data)
    out_trt = model_trt(data)

    assert(torch.allclose(out, out_trt, atol=1e-3, rtol=1e-3))