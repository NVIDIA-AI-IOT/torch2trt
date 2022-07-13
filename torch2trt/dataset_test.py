import pytest
import torch
import torch.nn as nn
from torch2trt.dataset import (
    TensorBatchDataset,
    ListDataset
)


def test_dataset_shapes():

    dataset = ListDataset()
    dataset.insert((torch.randn(1, 3, 32, 32), torch.randn(1, 4)))
    dataset.insert((torch.randn(1, 3, 64, 64), torch.randn(1, 8)))
    dataset.insert((torch.randn(1, 3, 48, 48), torch.randn(1, 6)))

    shapes = dataset.shapes()

    assert(torch.allclose(shapes[0][0], torch.LongTensor((1, 3, 32, 32))))
    assert(torch.allclose(shapes[0][1], torch.LongTensor((1, 3, 64, 64))))
    assert(torch.allclose(shapes[1][0], torch.LongTensor((1, 4))))
    assert(torch.allclose(shapes[1][1], torch.LongTensor((1, 8))))

    assert(torch.allclose(dataset.min_shapes()[0], torch.LongTensor((1, 3, 32, 32))))
    assert(torch.allclose(dataset.min_shapes()[1], torch.LongTensor((1, 4))))
    assert(torch.allclose(dataset.max_shapes()[0], torch.LongTensor((1, 3, 64, 64))))
    assert(torch.allclose(dataset.max_shapes()[1], torch.LongTensor((1, 8))))
    assert(torch.allclose(dataset.median_shapes()[0], torch.LongTensor((1, 3, 48, 48))))
    assert(torch.allclose(dataset.median_shapes()[1], torch.LongTensor((1, 6))))
    

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
