import pytest
import torch
import torch.nn as nn
from torch2trt import torch2trt, trt


class YOLOXFocusTestModule(nn.Module):


    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return x


def test_getitem_dynamic_yolox_layer():

    class YOLOXFocusTestModule(nn.Module):


        def forward(self, x):
            patch_top_left = x[..., ::2, ::2]
            patch_top_right = x[..., ::2, 1::2]
            patch_bot_left = x[..., 1::2, ::2]
            patch_bot_right = x[..., 1::2, 1::2]
            x = torch.cat(
                (
                    patch_top_left,
                    patch_bot_left,
                    patch_top_right,
                    patch_bot_right,
                ),
                dim=1,
            )
            return x

    module = YOLOXFocusTestModule().cuda().eval()

    data = torch.randn(1, 3, 112, 112).cuda()

    module_trt = torch2trt(module, [data], max_batch_size=4, log_level=trt.Logger.VERBOSE)

    data = torch.randn(1, 3, 112, 112).cuda()
    assert(torch.allclose(module_trt(data), module(data), atol=1e-4, rtol=1e-4))

    data = torch.randn(4, 3, 112, 112).cuda()
    assert(torch.allclose(module_trt(data), module(data), atol=1e-4, rtol=1e-4))


def test_getitem_dynamic_add_dim():

    class TestModule(nn.Module):


        def forward(self, x):
            patch_top_left = x[..., None]
            patch_top_right = x[..., None]
            patch_bot_left = x[..., None]
            patch_bot_right = x[..., None]
            x = torch.cat(
                (
                    patch_top_left,
                    patch_bot_left,
                    patch_top_right,
                    patch_bot_right,
                ),
                dim=1,
            )
            return x

    module = TestModule().cuda().eval()

    data = torch.randn(1, 3, 112, 112).cuda()

    module_trt = torch2trt(module, [data], max_batch_size=4, log_level=trt.Logger.VERBOSE)

    data = torch.randn(1, 3, 112, 112).cuda()
    assert(torch.allclose(module_trt(data), module(data), atol=1e-4, rtol=1e-4))

    data = torch.randn(4, 3, 112, 112).cuda()
    assert(torch.allclose(module_trt(data), module(data), atol=1e-4, rtol=1e-4))


def test_getitem_dynamic_remove_dim():

    class TestModule(nn.Module):


        def forward(self, x):
            patch_top_left = x[..., 0]
            patch_top_right = x[..., 0]
            patch_bot_left = x[..., 0]
            patch_bot_right = x[..., 0]
            x = torch.cat(
                (
                    patch_top_left,
                    patch_bot_left,
                    patch_top_right,
                    patch_bot_right,
                ),
                dim=1,
            )
            return x

    module = TestModule().cuda().eval()

    data = torch.randn(1, 3, 112, 112).cuda()

    module_trt = torch2trt(module, [data], max_batch_size=4, log_level=trt.Logger.VERBOSE)

    data = torch.randn(1, 3, 112, 112).cuda()
    assert(torch.allclose(module_trt(data), module(data), atol=1e-4, rtol=1e-4))

    data = torch.randn(4, 3, 112, 112).cuda()
    assert(torch.allclose(module_trt(data), module(data), atol=1e-4, rtol=1e-4))


def test_getitem_dynamic_remove_add_dim():

    class TestModule(nn.Module):


        def forward(self, x):
            patch_top_left = x[..., 0, None]
            patch_top_right = x[..., 0, None]
            patch_bot_left = x[..., 0, None]
            patch_bot_right = x[..., 0, None]
            x = torch.cat(
                (
                    patch_top_left,
                    patch_bot_left,
                    patch_top_right,
                    patch_bot_right,
                ),
                dim=1,
            )
            return x

    module = TestModule().cuda().eval()

    data = torch.randn(1, 3, 112, 112).cuda()

    module_trt = torch2trt(module, [data], max_batch_size=4, log_level=trt.Logger.VERBOSE)

    data = torch.randn(1, 3, 112, 112).cuda()
    assert(torch.allclose(module_trt(data), module(data), atol=1e-4, rtol=1e-4))

    data = torch.randn(4, 3, 112, 112).cuda()
    assert(torch.allclose(module_trt(data), module(data), atol=1e-4, rtol=1e-4))


def test_getitem_dynamic_gathernd():

    class TestModule(nn.Module):
        def forward(self, tensor, indices):
            #  indices = indices + indices
            return tensor[:, indices.to(torch.int64), (1, 0, 1)]

    module = TestModule().cuda().eval()

    tensor = torch.rand(3, 5, 4, 5).cuda()
    indices = torch.tensor((2, 0, 1), dtype=torch.int32).cuda()

    module_trt = torch2trt(module, [tensor, indices], min_shapes=[(1, 1, 1, 1), (1, 1, 1)], max_shapes=[(7, 7, 7, 7), (5, 5, 5)], log_level=trt.Logger.INFO)

    assert torch.allclose(module_trt(tensor, indices), module(tensor, indices), atol=1e-4, rtol=1e-4)

    tensor = torch.rand(2, 4, 3, 4).cuda()
    indices = torch.tensor((2, 0, 1), dtype=torch.int32).cuda()

    assert torch.allclose(module_trt(tensor, indices), module(tensor, indices), atol=1e-4, rtol=1e-4)

    tensor = torch.rand(4, 6, 5, 6).cuda()
    indices = torch.tensor((2, 0, 1), dtype=torch.int32).cuda()

    assert torch.allclose(module_trt(tensor, indices), module(tensor, indices), atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    torch.manual_seed(0)
    #  test_getitem_dynamic()
    test_getitem_dynamic_gathernd()
