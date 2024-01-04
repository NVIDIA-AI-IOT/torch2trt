import pytest
import torch
from torch2trt import torch2trt


def test_cpu_tracing():

    model = torch.nn.Conv2d(3, 3, kernel_size=1)

    data = torch.randn(1, 3, 32, 32)

    model_trt = torch2trt(model, [data])

    assert(hasattr(model_trt, 'engine'))
    assert(model_trt.engine is not None)

    data = torch.randn(1, 3, 32, 32)
    assert(torch.allclose(model(data), model_trt(data), atol=1e-3, rtol=1e-3))


if __name__ == '__main__':
    test_cpu_tracing()
