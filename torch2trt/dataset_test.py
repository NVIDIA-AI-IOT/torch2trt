import pytest
import torch
import torch.nn as nn
from torch2trt.dataset import (
    TensorBatchDataset,
    ListDataset,
    FolderDataset
)
from tempfile import mkdtemp


def test_dataset_shapes():

    dataset = ListDataset()
    dataset.insert((torch.randn(1, 3, 32, 32), torch.randn(1, 4)))
    dataset.insert((torch.randn(1, 3, 64, 64), torch.randn(1, 8)))
    dataset.insert((torch.randn(1, 3, 48, 48), torch.randn(1, 6)))

    shapes = dataset.shapes()

    assert(shapes[0][0] == (1, 3, 32, 32))
    assert(shapes[0][1] == (1, 3, 64, 64))
    assert(shapes[1][0] == (1, 4))
    assert(shapes[1][1] == (1, 8))

    assert(dataset.min_shapes()[0] == (1, 3, 32, 32))
    assert(dataset.min_shapes()[1] == (1, 4))
    assert(dataset.max_shapes()[0] == (1, 3, 64, 64))
    assert(dataset.max_shapes()[1] == (1, 8))
    assert(dataset.median_numel_shapes()[0] == (1, 3, 48, 48))
    assert(dataset.median_numel_shapes()[1] == (1, 6))


def test_dataset_infer_dynamic_axes():

    dataset = ListDataset()
    dataset.insert((torch.randn(1, 3, 32, 32), torch.randn(1, 4)))
    dataset.insert((torch.randn(1, 3, 64, 64), torch.randn(1, 8)))
    dataset.insert((torch.randn(1, 3, 48, 48), torch.randn(1, 6)))

    dynamic_axes = dataset.infer_dynamic_axes()
    
    assert(dynamic_axes[0] == [2, 3])
    assert(dynamic_axes[1] == [1])


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


def test_folder_dataset_record():

    dataset = FolderDataset(mkdtemp())

    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1).cuda().eval()

        def forward(self, x, y):
            a = self.conv(x)
            b = self.conv(y)
            return torch.cat([a, b], dim=0)

    device = torch.device('cuda:0')

    inputs = [
        torch.randn(1, 3, 32, 32, device=device),
        torch.randn(1, 3, 32, 32, device=device)
    ]

    module = TestModule().to(device).eval()

    with dataset.record(module):
        for i in range(5):
            module(*inputs)

    assert(len(dataset) == 5)
    assert(len(dataset[0]) == 2)
    assert(dataset[0][0].shape == (1, 3, 32, 32))
    assert(dataset[0][1].shape == (1, 3, 32, 32))
    assert(dataset[0][0].device == device)