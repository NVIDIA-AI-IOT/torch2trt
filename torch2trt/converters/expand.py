from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.Tensor.expand')
def convert_expand(ctx):
    input = ctx.method_args[0]
    sizes = ctx.method_args[1:]
    output = ctx.method_return
    
    inshape = tuple(input.shape)[1:] # exclude batch
    shape = tuple(output.shape)[1:]
    ndim = len(shape)
    start = tuple([0]*ndim)
    stride = tuple([int(i == o) for i, o in zip(inshape, shape)])  # stride == 1 if dimensions match, 0 otherwise
    
    layer = ctx.network.add_slice(input._trt, start, shape, stride)
    
    output._trt = layer.get_output(0)
    
    
class ExpandModule(torch.nn.Module):
    def __init__(self, *sizes):
        super(ExpandModule, self).__init__()
        self.sizes = sizes
        
    def forward(self, x):
        return x.expand(*self.sizes)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3,3)])
def test_tensor_expand_singledim():
    return ExpandModule(1, 3, 3, 3)
                                         
                                                    
@add_module_test(torch.float32, torch.device('cuda'), [(1,1,1,3)])
def test_tensor_expand_multidim():
    return ExpandModule(1, 3, 3, 3)
                                                       
                                                       
@add_module_test(torch.float32, torch.device('cuda'), [(1,1,1,3)])
def test_tensor_expand_inferdim():
    return ExpandModule(1, 3, -1, -1)