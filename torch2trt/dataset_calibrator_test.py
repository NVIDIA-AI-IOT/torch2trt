import pytest
import tensorrt as trt
import torch
import torch.nn as nn
from torch2trt.dataset import (
    TensorBatchDataset,
    ListDataset
)
from torch2trt import torch2trt


def test_dataset_calibrator_batch_dataset():

    torch.manual_seed(0)


    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1).cuda().eval()

        def forward(self, x, y):
            a = self.conv(x)
            b = self.conv(y)
            return torch.cat([a, b], dim=0)

    inputs = [
        torch.randn(1, 3, 32, 32).cuda(),
        torch.randn(1, 3, 32, 32).cuda()
    ]

    module = TestModule().cuda().eval()

    dataset = TensorBatchDataset()

    with dataset.record(module):
        for i in range(50):
            module(*inputs)
    
    module_trt = torch2trt(
        module,
        dataset[0],
        int8_mode=True,
        int8_calib_dataset=dataset,
        log_level=trt.Logger.INFO
    )

    inputs = [
        torch.randn(1, 3, 32, 32).cuda(),
        torch.randn(1, 3, 32, 32).cuda()
    ]
    output = module(*inputs)
    output_trt = module_trt(*inputs)

    assert(torch.allclose(output, output_trt, rtol=1e-3, atol=1e-3))


def test_dataset_calibrator_list_dataset():

    torch.manual_seed(0)


    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1).cuda().eval()

        def forward(self, x, y):
            a = self.conv(x)
            b = self.conv(y)
            return torch.cat([a, b], dim=0)

    inputs = [
        torch.randn(1, 3, 32, 32).cuda(),
        torch.randn(1, 3, 32, 32).cuda()
    ]

    module = TestModule().cuda().eval()

    dataset = ListDataset()

    with dataset.record(module):
        for i in range(50):
            module(*inputs)
    
    module_trt = torch2trt(
        module,
        dataset[0],
        int8_mode=True,
        int8_calib_dataset=dataset,
        log_level=trt.Logger.INFO
    )

    inputs = [
        torch.randn(1, 3, 32, 32).cuda(),
        torch.randn(1, 3, 32, 32).cuda()
    ]
    output = module(*inputs)
    output_trt = module_trt(*inputs)

    assert(torch.allclose(output, output_trt, rtol=1e-3, atol=1e-3))


if __name__ == '__main__':
    test_dataset_calibrator_list_dataset()