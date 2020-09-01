from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


def __add_clamp(network, trt_input, val, op):
    
    # create TensorRT constant for minimum value
    val_shape = (1, ) * len(trt_input.shape)  # broadcast all dimensions
    val_tensor = val * torch.ones(val_shape, dtype=torch_dtype_from_trt(trt_input.dtype)).cpu().numpy()
    val_trt = network.add_constant(val_shape, val_tensor)
    layer = network.add_elementwise(trt_input, val_trt.get_output(0), op)
    
    return layer

    
# CLAMP_MIN

    
@tensorrt_converter('torch.clamp_min')
@tensorrt_converter('torch.Tensor.clamp_min')
def convert_clamp_min(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    val = ctx.method_args[1]
    output = ctx.method_return
    
    layer = __add_clamp(ctx.network, input_trt, val, trt.ElementWiseOperation.MAX)
    
    output._trt = layer.get_output(0)

    
class TorchClampMin(torch.nn.Module):
    def forward(self, x):
        return torch.clamp_min(x, -0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_torch_clamp_min():
    return TorchClampMin()


class TensorClampMin(torch.nn.Module):
    def forward(self, x):
        return x.clamp_min(-0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_tensor_clamp_min():
    return TensorClampMin()

    
# CLAMP_MAX


@tensorrt_converter('torch.clamp_max')
@tensorrt_converter('torch.Tensor.clamp_max')
def convert_clamp_max(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    val = ctx.method_args[1]
    output = ctx.method_return
    
    layer = __add_clamp(ctx.network, input_trt, val, trt.ElementWiseOperation.MIN)
    
    output._trt = layer.get_output(0)
    

class TorchClampMax(torch.nn.Module):
    def forward(self, x):
        return torch.clamp_max(x, 0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_torch_clamp_max():
    return TorchClampMax()


class TensorClampMax(torch.nn.Module):
    def forward(self, x):
        return x.clamp_max(0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_tensor_clamp_max():
    return TensorClampMax()


# CLAMP
    
@tensorrt_converter('torch.clamp')
@tensorrt_converter('torch.Tensor.clamp')
def convert_clamp(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    if "min" in ctx.method_kwargs and "max" in ctx.method_kwargs:
        min_val = ctx.method_kwargs["min"]
        max_val = ctx.method_kwargs["max"]
        layer = __add_clamp(ctx.network, input_trt, min_val, trt.ElementWiseOperation.MAX)
        layer = __add_clamp(ctx.network, layer.get_output(0), max_val, trt.ElementWiseOperation.MIN)
    elif "min" in ctx.method_kwargs:
        min_val = ctx.method_kwargs["min"]
        layer = __add_clamp(ctx.network, input_trt, min_val, trt.ElementWiseOperation.MAX)
    elif "max" in ctx.method_kwargs:
        max_val = ctx.method_kwargs["max"]
        layer = __add_clamp(ctx.network, input_trt, max_val, trt.ElementWiseOperation.MIN)
    else:
        min_val = ctx.method_args[1]
        max_val = ctx.method_args[2]
        layer = __add_clamp(ctx.network, input_trt, min_val, trt.ElementWiseOperation.MAX)
        layer = __add_clamp(ctx.network, layer.get_output(0), max_val, trt.ElementWiseOperation.MIN)
    
    output._trt = layer.get_output(0)
    

class TorchClamp(torch.nn.Module):
    def forward(self, x):
        return torch.clamp(x, -0.1, 0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_torch_clamp():
    return TorchClamp()


class TensorClamp(torch.nn.Module):
    def forward(self, x):
        return x.clamp(-0.1, 0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_tensor_clamp():
    return TensorClamp()


class TorchClampOptionMax(torch.nn.Module):
    def forward(self, x):
        return torch.clamp(x, max=0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_torch_clamp_option_max():
    return TorchClampOptionMax()

class TorchClampOptionMin(torch.nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=-0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_torch_clamp_option_min():
    return TorchClampOptionMin()


class TorchClampOptionMaxMin(torch.nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=-0.1, max=0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_torch_clamp_option_max_min():
    return TorchClampOptionMaxMin()


class TensorClampOptionMax(torch.nn.Module):
    def forward(self, x):
        return x.clamp(max=0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_tensor_clamp_option_max():
    return TensorClampOptionMax()

class TensorClampOptionMin(torch.nn.Module):
    def forward(self, x):
        return x.clamp(min=-0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_tensor_clamp_option_min():
    return TensorClampOptionMin()


class TensorClampOptionMaxMin(torch.nn.Module):
    def forward(self, x):
        return x.clamp(min=-0.1, max=0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_tensor_clamp_option_max_min():
    return TensorClampOptionMaxMin()