from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


def slice_to_trt(dim_size, dim_slice):
    
    start = 0 if dim_slice.start is None else dim_slice.start
    stop = dim_size if dim_slice.stop is None else dim_slice.stop
    stride = 1 if dim_slice.step is None else dim_slice.step
    
    size = (stop - start - 1) // stride + 1
    
    return start, size, stride


def num_slice_types(slices):
    num_slice = 0
    for s in slices:
        if isinstance(s, slice) or isinstance(s, int):
            num_slice += 1
    return num_slice


@tensorrt_converter('torch.Tensor.__getitem__')
def convert_tensor_getitem(ctx):
    input = ctx.method_args[0]
    slices = ctx.method_args[1]
    output = ctx.method_return
    
    input_trt = input._trt
    
    # Step 1 - Replace ellipsis with expanded slices
    
    num_ellipsis = input.ndim - num_slice_types(slices)
    
    new_slices = []
    for s in slices:
        
        if s == Ellipsis:
            while num_ellipsis > 0:
                new_slices.append(slice(None, None, None))
                num_ellipsis -= 1
        elif isinstance(s, slice):
            new_slices.append(s)
        elif s is None:
            new_slices.append(None)
        elif isinstance(s, int):
            new_slices.append(s)
            
    # fill missing slices at end
    while num_slice_types(new_slices) < len(input.shape):
        new_slices.append(slice(None, None, None))
            
    # Step 2 - Remove batch from slices (TRT from this point)
    
    slices = tuple(new_slices[1:]) # remove batch
    
    
    # Step 3 - Add slice layer (will currently ignore 'None' slices)
    
    starts = []
    sizes = []
    strides = []
    
    input_dim = 0
    for s in slices:
        
        if input_dim >= len(input_trt.shape):
            break
            
        input_size = int(input_trt.shape[input_dim])
        
        if isinstance(s, slice):
            start, size, stride = slice_to_trt(input_size, s)
            starts.append(start)
            sizes.append(size)
            strides.append(stride)
            input_dim += 1
            
        elif isinstance(s, int):
            starts.append(s)
            sizes.append(1)
            strides.append(1)
            input_dim += 1
    
    output_trt = ctx.network.add_slice(input_trt, starts, sizes, strides).get_output(0)
    
    # Step 4 - Add shuffle layer to insert dimensions for 'None' slices and remove dimensions for 'int' slices
    
    num_non_slice = len([s for s in slices if not isinstance(s, slice)])
    if num_non_slice > 0:
        layer = ctx.network.add_shuffle(output_trt)
        layer.reshape_dims = tuple(output.shape[1:]) # exclude batch
        output_trt = layer.get_output(0)
        
    output._trt = output_trt
    
    
class LambdaModule(torch.nn.Module):
    def __init__(self, fn):
        super(LambdaModule, self).__init__()
        self.fn = fn
        
    def forward(self, x):
        return self.fn(x)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_tensor_getitem_1d_int():
    return LambdaModule(lambda x: x[:, 0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_int():
    return LambdaModule(lambda x: x[:, 0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_strided():
    return LambdaModule(lambda x: x[:, ::2])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_strided_offset():
    return LambdaModule(lambda x: x[:, 1::2])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_strided_range():
    return LambdaModule(lambda x: x[:, 1:3:2])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_insert_dim():
    return LambdaModule(lambda x: x[:, None])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_insert_dim_ellipsis():
    return LambdaModule(lambda x: x[:, None, ...])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_append_dim():
    return LambdaModule(lambda x: x[:, ..., None])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_append_2dim():
    return LambdaModule(lambda x: x[:, ..., None, None])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_weird_combo():
    return LambdaModule(lambda x: x[:, 0:3:4, None, None, 1, ...])