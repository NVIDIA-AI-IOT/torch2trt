import pytest
import torch
import torch.nn as nn
from torch2trt.dataset import (
    TensorBatchDataset,
    ListDataset
)


def test_tensor_batch_dataset_record():

    dataset = TensorBatchDataset()

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

    with dataset.record(module):
        for i in range(5):
            module(*inputs)

    assert(len(dataset) == 5)
    assert(len(dataset[0]) == 2)
    assert(dataset[0][0].shape == (1, 3, 32, 32))
    assert(dataset[0][1].shape == (1, 3, 32, 32))


def test_list_dataset_record():

    dataset = ListDataset()

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

    with dataset.record(module):
        for i in range(5):
            module(*inputs)

    assert(len(dataset) == 5)
    assert(len(dataset[0]) == 2)
    assert(dataset[0][0].shape == (1, 3, 32, 32))
    assert(dataset[0][1].shape == (1, 3, 32, 32))
