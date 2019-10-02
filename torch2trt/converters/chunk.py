from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from .split import convert_split


@tensorrt_converter('torch.chunk')
@tensorrt_converter('torch.Tensor.chunk')
def convert_chunk(ctx):
    convert_split(ctx)

        
class TorchChunk(torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(TorchChunk, self).__init__()
        self.args = args
        self.kwargs = kwargs
        
    def forward(self, x):
        return torch.chunk(x, *self.args, **self.kwargs)
    

class TensorChunk(torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(TensorChunk, self).__init__()
        self.args = args
        self.kwargs = kwargs
        
    def forward(self, x):
        return x.chunk(*self.args, **self.kwargs)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_chunk_1_1():
    return TorchChunk(1, 1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_chunk_2_1():
    return TorchChunk(2, 1)

    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_chunk_3_1():
    return TorchChunk(3, 1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_chunk_3_2():
    return TorchChunk(3, 2)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_tensor_chunk_3_2():
    return TensorChunk(3, 2)