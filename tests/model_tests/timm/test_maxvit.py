import pytest
import torch2trt
from timm.models.maxxvit import (
    maxvit_tiny_rw_224,
    maxvit_rmlp_pico_rw_256,
    maxvit_rmlp_small_rw_224
)
import torch


def _cross_validate_module(model, shape=(224, 224)):
    model = model.cuda()
    data = torch.randn(1, 3, *shape).cuda()
    model_trt = torch2trt.torch2trt(model, [data])
    out = model(data)
    out_trt = model_trt(data)
    assert torch.allclose(out, out_trt, rtol=1e-2, atol=1e-2)


def test_maxvit_tiny_rw_224():
    _cross_validate_module(maxvit_tiny_rw_224().cuda().eval(), (224, 224))


def test_maxvit_rmlp_small_rw_224():
    _cross_validate_module(maxvit_rmlp_small_rw_224().cuda().eval(), (224, 224))


if __name__ == "__main__":
    test_maxvit_tiny_rw_224()