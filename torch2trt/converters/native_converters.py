import math
from torch2trt.torch2trt import *
from torch2trt.version_utils import Version
import sys

if Version(sys.version.split(" ")[0]) >= "3.10":
    import collections.abc as collections
else:
    import collections


@tensorrt_converter('torch.nn.functional.leaky_relu')
@tensorrt_converter('torch.nn.functional.leaky_relu_')
def convert_leaky_relu(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    negative_slope = get_arg(ctx, 'negative_slope', pos=1, default=0.01)
    output = ctx.method_return
    
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.LEAKY_RELU)
    layer.alpha = negative_slope
    
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.nn.functional.elu')
@tensorrt_converter('torch.nn.functional.elu_')
def convert_elu(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    alpha = get_arg(ctx, 'alpha', pos=1, default=1.0)
    output = ctx.method_return
    
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.ELU)
    layer.alpha = alpha
    
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.selu')
@tensorrt_converter('torch.selu_')
@tensorrt_converter('torch.nn.functional.selu')
@tensorrt_converter('torch.nn.functional.selu_')
def convert_selu(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    alpha = get_arg(ctx, 'alpha', pos=1, default=1.0)
    output = ctx.method_return
    
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SELU)
    layer.alpha = 1.6732632423543772848170429916717
    layer.beta = 1.0507009873554804934193349852946
    
    output._trt = layer.get_output(0)
 

@tensorrt_converter('torch.nn.functional.softsign')
def convert_softsign(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    output = ctx.method_return
    
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SOFTSIGN)
    
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.nn.functional.softplus')
def convert_softplus(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    output = ctx.method_return
    
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SOFTPLUS)
    
    output._trt = layer.get_output(0)
   

def convert_adaptive_pool(ctx, pool_type: trt.PoolingType):
    input = ctx.method_args[0]
    output_size = ctx.method_args[1]
    output = ctx.method_return

    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    nd = input.ndim - 2

    if nd == 1:
        raise NotImplementedError
        
    if not isinstance(output_size, tuple):
        output_size = (output_size,) * nd

    stride = []
    for i in range(nd):
        idx = i - nd
        stride.append(input.shape[idx] // output_size[idx])

    kernel_size = stride
    layer = ctx.network.add_pooling_nd(
        input=input_trt,
        type=pool_type,
        window_size=kernel_size,
    )
    layer.stride_nd = stride

    output._trt = layer.get_output(0)


@tensorrt_converter('torch.nn.functional.adaptive_avg_pool1d')
@tensorrt_converter('torch.nn.functional.adaptive_avg_pool2d')
@tensorrt_converter('torch.nn.functional.adaptive_avg_pool3d')
def convert_adaptive_avg_pool(ctx):
    convert_adaptive_pool(ctx, trt.PoolingType.AVERAGE)


@tensorrt_converter('torch.nn.functional.adaptive_max_pool1d')
@tensorrt_converter('torch.nn.functional.adaptive_max_pool2d')
@tensorrt_converter('torch.nn.functional.adaptive_max_pool3d')
def convert_adaptive_max_pool(ctx):
    convert_adaptive_pool(ctx, trt.PoolingType.MAX)


@tensorrt_converter('torch.add')
@tensorrt_converter('torch.Tensor.__iadd__')
@tensorrt_converter('torch.Tensor.__add__')
@tensorrt_converter('torch.Tensor.__radd__')
def convert_add(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.SUM)
    output._trt = layer.get_output(0)




@tensorrt_converter('torch.nn.functional.batch_norm')
def convert_batch_norm(ctx):

    input = get_arg(ctx, 'input', pos=0, default=None) 
    running_mean = get_arg(ctx, 'running_mean', pos=1, default=None) 
    running_var = get_arg(ctx, 'running_var', pos=2, default=None) 

    weight = get_arg(ctx, 'weight', pos=3, default=None) 
    bias = get_arg(ctx, 'bias', pos=4, default=None) 
    eps = get_arg(ctx, 'eps', pos=7, default=10e-6) 

    ndim = input.ndim - 2
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    
    scale = weight.detach().cpu().numpy() / np.sqrt(running_var.detach().cpu().numpy() + eps)
    bias = bias.detach().cpu().numpy() - running_mean.detach().cpu().numpy() * scale
    power = np.ones_like(scale)

    if ndim == 1:
        # reshape to 2D
        layer = ctx.network.add_shuffle(input_trt)
        
        if len(input.shape) == 2:
            layer.reshape_dims = (0, 0, 1, 1)
        else:
            layer.reshape_dims = (0, 0, 0, 1)

        scale_input = layer.get_output(0)
    else:
        scale_input = input_trt

    layer = ctx.network.add_scale_nd(scale_input, trt.ScaleMode.CHANNEL, bias, scale, power, 1)

    if ndim == 1:
        # reshape back to 1D
        layer = ctx.network.add_shuffle(layer.get_output(0))
        if len(input.shape) == 2:
            layer.reshape_dims = (0, 0)
        else:
            layer.reshape_dims = (0, 0, 0)


    output._trt = layer.get_output(0)


@tensorrt_converter('torch.cat')
def convert_cat(ctx):
    inputs = get_arg(ctx, 'input', pos=0, default=None)
    dim = get_arg(ctx, 'dim', pos=1, default=0)

    # Reverse negative dims.
    if dim < 0:
        dim = len(inputs[0].shape) - abs(dim)

    output = ctx.method_return
    trt_inputs = add_missing_trt_tensors(ctx.network, inputs)
    trt_inputs = broadcast_trt_tensors(ctx.network, trt_inputs, len(output.shape))

    layer = ctx.network.add_concatenation(inputs=trt_inputs)
    layer.axis = dim
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.chunk')
@tensorrt_converter('torch.Tensor.chunk')
@tensorrt_converter('torch.split')
@tensorrt_converter('torch.Tensor.split')
def convert_split_or_chunk(ctx):
    input = get_arg(ctx, 'input', 0, None)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    # we don't need to parse split/chunk (arg 1)
    # since we infer size from output tensors
    dim = get_arg(ctx, 'dim', 2, 0)
    
    outputs = ctx.method_return
    
    # assert(dim >= 1)
    
    start = [0] * len(input.shape) 
    stride = [1] * len(start)
    offset = 0
    
    # add slice layers
    for i, output in enumerate(outputs):
        shape = list(output.shape) 
        start[dim] = offset
        layer = ctx.network.add_slice(input_trt, start=start, shape=shape, stride=stride)
        output._trt = layer.get_output(0)
        offset = offset + shape[dim]


def _add_clamp_val(network, trt_input, val, op):
    # create TensorRT constant for minimum value
    val_shape = (1, ) * len(trt_input.shape)  # broadcast all dimensions
    val_tensor = val * torch.ones(val_shape, dtype=torch_dtype_from_trt(trt_input.dtype)).cpu().numpy()
    val_trt = network.add_constant(val_shape, val_tensor)
    layer = network.add_elementwise(trt_input, val_trt.get_output(0), op)

    return layer


def _add_clamp_tensor(network, trt_input, tensor, op):
    tensor_trt = trt_(network, tensor)
    trt_input, tensor_trt = broadcast_trt_tensors(network, [trt_input, tensor_trt], max(len(trt_input.shape), len(tensor_trt.shape)))
    layer = network.add_elementwise(trt_input, tensor_trt, op)

    return layer


def __add_clamp(network, trt_input, val, op):
    return (_add_clamp_tensor(network, trt_input, val, op) if isinstance(val, torch.Tensor)
        else _add_clamp_val(network, trt_input, val, op))


@tensorrt_converter('torch.clamp_min')
@tensorrt_converter('torch.Tensor.clamp_min')
def convert_clamp_min(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    val = ctx.method_args[1]
    output = ctx.method_return
    
    layer = __add_clamp(ctx.network, input_trt, val, trt.ElementWiseOperation.MAX)
    
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.clamp_max')
@tensorrt_converter('torch.Tensor.clamp_max')
def convert_clamp_max(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    val = ctx.method_args[1]
    output = ctx.method_return
    
    layer = __add_clamp(ctx.network, input_trt, val, trt.ElementWiseOperation.MIN)
    
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.clamp')
@tensorrt_converter('torch.Tensor.clamp')
def convert_clamp(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    min_val = get_arg(ctx, "min", 1, None)
    max_val = get_arg(ctx, "max", 2, None)
    if min_val is not None and max_val is not None:
        layer = __add_clamp(ctx.network, input_trt, min_val, trt.ElementWiseOperation.MAX)
        layer = __add_clamp(ctx.network, layer.get_output(0), max_val, trt.ElementWiseOperation.MIN)
    elif min_val is not None:
        layer = __add_clamp(ctx.network, input_trt, min_val, trt.ElementWiseOperation.MAX)
    elif max_val is not None:
        layer = __add_clamp(ctx.network, input_trt, max_val, trt.ElementWiseOperation.MIN)
    else:
        raise RuntimeError("Unsupported argument combination")
    
    output._trt = layer.get_output(0)
    

@tensorrt_converter('torch.clone')
@tensorrt_converter('torch.Tensor.clone')
def convert_clone(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)

    # Clone by making identity layer.
    layer = ctx.network.add_identity(input_trt)
    set_layer_precision(ctx, layer)

    output = ctx.method_return
    output._trt = layer.get_output(0)

def _convert_elementwise(ctx, op):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], max(len(input_a_trt.shape), len(input_b_trt.shape)))
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, op)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.gt')
@tensorrt_converter('torch.Tensor.__gt__')
def convert_gt(ctx):
    return _convert_elementwise(ctx, trt.ElementWiseOperation.GREATER)


@tensorrt_converter('torch.lt')
@tensorrt_converter('torch.Tensor.__lt__')
def convert_gt(ctx):
    return _convert_elementwise(ctx, trt.ElementWiseOperation.LESS)


@tensorrt_converter('torch.eq')
@tensorrt_converter('torch.Tensor.__eq__')
def convert_gt(ctx):
    return _convert_elementwise(ctx, trt.ElementWiseOperation.EQUAL)



@tensorrt_converter('torch.nn.functional.conv1d')
@tensorrt_converter('torch.nn.functional.conv2d')
@tensorrt_converter('torch.nn.functional.conv3d')
def convert_conv2d3d(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    weight = get_arg(ctx, 'weight', pos=1, default=None)
    bias = get_arg(ctx, 'bias', pos=2, default=None)
    stride = get_arg(ctx, 'stride', pos=3, default=1)
    padding = get_arg(ctx, 'padding', pos=4, default=0)
    dilation = get_arg(ctx, 'dilation', pos=5, default=1)
    groups = get_arg(ctx, 'groups', pos=6, default=1)
    
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    input_dim = input.dim() - 2
    
    out_channels = int(weight.shape[0])
    kernel_size = tuple(weight.shape[2:])
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * input_dim

    if not isinstance(stride, tuple):
        stride = (stride, ) * input_dim

    if not isinstance(padding, tuple):
        padding = (padding, ) * input_dim

    if not isinstance(dilation, tuple):
        dilation = (dilation, ) * input_dim


    kernel = weight.detach().cpu().numpy()
    
    if bias is not None:
        bias = bias.detach().cpu().numpy()

    # Handle reshape 1D to 2D
    if input_dim == 1:
        kernel_size = kernel_size + (1,)
        stride = stride + (1,)
        padding = padding + (0,)
        dilation = dilation + (1,)
        unsqueeze_layer = ctx.network.add_shuffle(input_trt)
        set_layer_precision(ctx, unsqueeze_layer)
        unsqueeze_layer.reshape_dims = tuple([0]*input.ndim) + (1,) 
        conv_input = unsqueeze_layer.get_output(0)
    else:
        conv_input = input_trt


    conv_layer = ctx.network.add_convolution_nd(
        input=conv_input,
        num_output_maps=out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias)
    conv_layer.stride_nd = stride
    conv_layer.padding_nd = padding
    conv_layer.dilation_nd = dilation

    if groups is not None:
        conv_layer.num_groups = groups

    output._trt = conv_layer.get_output(0)

    # Handle reshape 2D backt o 1D
    if input_dim == 1:
        squeeze_layer = ctx.network.add_shuffle(conv_layer.get_output(0))
        set_layer_precision(ctx, squeeze_layer)
        squeeze_layer.reshape_dims = tuple([0] * input.ndim)
        output._trt = squeeze_layer.get_output(0)
    else:
        output._trt = conv_layer.get_output(0)



@tensorrt_converter('torch.nn.functional.conv_transpose1d')
@tensorrt_converter('torch.nn.functional.conv_transpose2d')
@tensorrt_converter('torch.nn.functional.conv_transpose3d')
def convert_conv_transpose2d3d(ctx):

    input = get_arg(ctx, 'input', pos=0, default=None)
    weight = get_arg(ctx, 'weight', pos=1, default=None)
    bias = get_arg(ctx, 'bias', pos=2, default=None)
    stride = get_arg(ctx, 'stride', pos=3, default=1)
    padding = get_arg(ctx, 'padding', pos=4, default=0)
    dilation = get_arg(ctx, 'dilation', pos=5, default=1)
    groups = get_arg(ctx, 'groups', pos=6, default=1)
    
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    input_dim = input.dim() - 2
    
    out_channels = int(weight.shape[0])
    kernel_size = tuple(weight.shape[2:])

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * input_dim

    if not isinstance(stride, tuple):
        stride = (stride, ) * input_dim

    if not isinstance(padding, tuple):
        padding = (padding, ) * input_dim

    if not isinstance(dilation, tuple):
        dilation = (dilation, ) * input_dim

    kernel = weight.detach().cpu().numpy()
    

    if bias is not None:
        bias = bias.detach().cpu().numpy()
    else:

        bias = trt.Weights(torch_dtype_to_trt(weight.dtype))


    # Handle reshape 1D to 2D
    if input_dim == 1:
        kernel_size = kernel_size + (1,)
        stride = stride + (1,)
        padding = padding + (0,)
        dilation = dilation + (1,)
        unsqueeze_layer = ctx.network.add_shuffle(input_trt)
        set_layer_precision(ctx, unsqueeze_layer)
        unsqueeze_layer.reshape_dims = tuple([0]*input.ndim) + (1,) 
        conv_input = unsqueeze_layer.get_output(0)
    else:
        conv_input = input_trt


    conv_layer = ctx.network.add_deconvolution_nd(
        input=conv_input,
        num_output_maps=out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias)
    conv_layer.stride_nd = stride
    conv_layer.padding_nd = padding
    
    if groups is not None:
        conv_layer.num_groups = groups


    # Handle reshape 2D backt o 1D
    if input_dim == 1:
        squeeze_layer = ctx.network.add_shuffle(conv_layer.get_output(0))
        set_layer_precision(ctx, squeeze_layer)
        squeeze_layer.reshape_dims = tuple([0] * input.ndim)
        output._trt = squeeze_layer.get_output(0)
    else:
        output._trt = conv_layer.get_output(0)



@tensorrt_converter('torch.div')
@tensorrt_converter('torch.Tensor.__div__') # py2
@tensorrt_converter('torch.Tensor.__idiv__') # py2
@tensorrt_converter('torch.Tensor.__truediv__') # py3
@tensorrt_converter('torch.Tensor.__itruediv__') # py3
def convert_div(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.DIV)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.Tensor.__rdiv__') # py2
@tensorrt_converter('torch.Tensor.__rtruediv__') # py3
def convert_rdiv(ctx):
    input_a = ctx.method_args[1]  # inputs switched for rdiv
    input_b = ctx.method_args[0]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.DIV)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.einsum')
def convert_einsum(ctx):
    einsum_eq = ctx.method_args[0]
    input_tensors = ctx.method_args[1:]
    output = ctx.method_return
    
    layer = ctx.network.add_einsum(
        [t._trt for t in input_tensors],
        einsum_eq
    )

    output._trt = layer.get_output(0)


@tensorrt_converter('torch.Tensor.expand')
def convert_expand(ctx):
    input = ctx.method_args[0]

    if not hasattr(input, '_trt'):
        return
        
    sizes = ctx.method_args[1:]
    output = ctx.method_return
    
    inshape = tuple(input.shape)
    shape = tuple(output.shape)
    ndim = len(shape)
    start = tuple([0]*ndim)
    stride = tuple([int(i == o) for i, o in zip(inshape, shape)])  # stride == 1 if dimensions match, 0 otherwise
    
    layer = ctx.network.add_slice(input._trt, start, shape, stride)
    
    output._trt = layer.get_output(0)
    

@tensorrt_converter('torch.flatten')
@tensorrt_converter('torch.Tensor.flatten')
def convert_flatten(ctx):
    input = ctx.method_args[0]
    start_dim = get_arg(ctx, 'start_dim', pos=1, default=0)
    end_dim = get_arg(ctx, 'end_dim', pos=2, default=-1)

    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    start_dim = torch_dim_resolve_negative(start_dim, input.ndim)[0]
    end_dim = torch_dim_resolve_negative(end_dim, input.ndim)[0]

    input_shape_trt = ctx.network.add_shape(input_trt).get_output(0)
    new_shape_trt = []

    # get shape before flatten
    for i in range(start_dim):
        dim_trt = ctx.network.add_slice(input_shape_trt, [i], [1], [1]).get_output(0)
        new_shape_trt.append(dim_trt)

    # get flatten reduce dimensions
    if start_dim != end_dim:
        new_shape_trt.append(
            ctx.network.add_constant([1], np.array([-1], dtype=trt_int_dtype())).get_output(0)
        )

    # get shape after flatten
    for i in range(end_dim + 1, input.ndim):
        dim_trt = ctx.network.add_slice(input_shape_trt, [i], [1], [1]).get_output(0)
        new_shape_trt.append(dim_trt)

    new_shape_trt = ctx.network.add_concatenation(new_shape_trt).get_output(0)

    # do flatten with shuffle layer
    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, new_shape_trt)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.Tensor.__floordiv__')
@tensorrt_converter('torch.Tensor.__ifloordiv__')
@tensorrt_converter('torch.floor_divide')
def convert_floordiv(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))
    # we can not use ElementWiseOperation.FLOOR_DIV directly because Torch truncate negative result toward 0
    # but TensorRT FLOOR_DIV op toward -Inf
    # sign = ab / |ab|
    # floordiv result: sign * (|a| // |b|)
    ab_layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.PROD)
    abs_ab_layer = ctx.network.add_unary(ab_layer.get_output(0), trt.UnaryOperation.ABS)
    sign_layer = ctx.network.add_elementwise(ab_layer.get_output(0), abs_ab_layer.get_output(0),
                                             trt.ElementWiseOperation.DIV)
    abs_a_layer = ctx.network.add_unary(input_a_trt, trt.UnaryOperation.ABS)
    abs_b_layer = ctx.network.add_unary(input_b_trt, trt.UnaryOperation.ABS)
    abs_floor_layer = ctx.network.add_elementwise(abs_a_layer.get_output(0), abs_b_layer.get_output(0),
                                                  trt.ElementWiseOperation.FLOOR_DIV)
    out_layer = ctx.network.add_elementwise(sign_layer.get_output(0), abs_floor_layer.get_output(0),
                                            trt.ElementWiseOperation.PROD)
    output._trt = out_layer.get_output(0)


@tensorrt_converter('torch.nn.functional.gelu')
def convert_gelu(ctx):
    # approximate equation 1 from paper
    input = get_arg(ctx, 'input', 0, None)
    output = ctx.method_return
    
    x, c05, c1, cs2pi, c044, c3 = add_missing_trt_tensors(
        ctx.network,
        [input, 0.5, 1.0, math.sqrt(2.0 / math.pi), 0.044715, 3.0]
    )
    
    x, c05, c1, cs2pi, c044, c3 = broadcast_trt_tensors(
        ctx.network, 
        [x, c05, c1, cs2pi, c044, c3], 
        len(output.shape)
    )
    
    y = ctx.network.add_elementwise(x, c3, trt.ElementWiseOperation.POW).get_output(0)
    y = ctx.network.add_elementwise(y, c044, trt.ElementWiseOperation.PROD).get_output(0)
    y = ctx.network.add_elementwise(x, y, trt.ElementWiseOperation.SUM).get_output(0)
    y = ctx.network.add_elementwise(y, cs2pi, trt.ElementWiseOperation.PROD).get_output(0)
    y = ctx.network.add_activation(y, trt.ActivationType.TANH).get_output(0)
    y = ctx.network.add_elementwise(y, c1, trt.ElementWiseOperation.SUM).get_output(0)
    y = ctx.network.add_elementwise(x, y, trt.ElementWiseOperation.PROD).get_output(0)
    y = ctx.network.add_elementwise(y, c05, trt.ElementWiseOperation.PROD).get_output(0)
    
    output._trt = y


def slice_to_trt(ctx, dim_size, dim_slice):
    
    start = 0 if dim_slice.start is None else dim_slice.start
    stop = dim_size if dim_slice.stop is None else dim_slice.stop
    stride = 1 if dim_slice.step is None else dim_slice.step
    
    start = make_int_wrapper(start)
    stop = make_int_wrapper(stop)
    stride = make_int_wrapper(stride)

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
    
    if not hasattr(input, '_trt'):
        return

    input_trt = input._trt
    
    # Step 1 - Replace ellipsis with expanded slices
    
    num_ellipsis = len(input.shape) - num_slice_types(slices)
    
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
        elif isinstance(s, int) or isinstance(s, IntWrapper):
            new_slices.append(s)
            
    # fill missing slices at end
    while num_slice_types(new_slices) < len(input.shape):
        new_slices.append(slice(None, None, None))
            
    # Step 2 - Remove batch from slices (TRT from this point)
    
    slices = tuple(new_slices) # remove batch
    
    
    # Step 3 - Add slice layer (will currently ignore 'None' slices)
    
    starts = []
    sizes = []
    strides = []
    
    input_dim = 0

    input_size = input.size()

    for s in slices:
        
        if input_dim >= len(input_trt.shape):
            break
        
        if isinstance(s, slice):
            start, size, stride = slice_to_trt(ctx, input_size[input_dim], s)
            starts.append(start)
            sizes.append(size)
            strides.append(stride)
            input_dim += 1
            
        elif isinstance(s, int) or isinstance(s, IntWrapper):
            starts.append(make_int_wrapper(s))
            sizes.append(make_int_wrapper(1))
            strides.append(make_int_wrapper(1))
            input_dim += 1
    
    starts = make_size_wrapper(starts)
    sizes = make_size_wrapper(sizes)
    strides = make_size_wrapper(strides)
    
    # make positive
    def make_positive(size):
        sizes = []
        for i in range(len(size)):
            size_i = size[i]
            if size_i < 0:
                input_size_i = input_size[i]
                size_i = input_size_i + size_i
            sizes.append(size_i)
        return make_size_wrapper(sizes)

    starts = make_positive(starts)
    sizes = make_positive(sizes)
    strides = make_positive(strides)

    layer = ctx.network.add_slice(input_trt, starts, sizes, strides)
    layer.set_input(1, starts._trt)
    layer.set_input(2, sizes._trt)
    layer.set_input(3, strides._trt)

    output_trt = layer.get_output(0)

    # Step 4 - Add shuffle layer to insert dimensions for 'None' slices and remove dimensions for 'int' slices


    num_non_slice = len([s for s in slices if not isinstance(s, slice)])
    if num_non_slice > 0:

        final_shape = []
        i = 0
        for s in slices:
            if isinstance(s, slice):
                # copy slice dim
                final_shape.append(sizes[i])
                i += 1
            elif isinstance(s, int) or isinstance(s, IntWrapper):
                # remove int dim
                i += 1  
            else:
                # insert None dim
                final_shape.append(make_int_wrapper(1))
                
        final_shape = make_size_wrapper(final_shape)

        layer = ctx.network.add_shuffle(output_trt)
        layer.reshape_dims = tuple(output.shape) # exclude batch
        layer.set_input(1, final_shape._trt)
        output_trt = layer.get_output(0)
        
    output._trt = output_trt
    

@tensorrt_converter('torch.nn.functional.group_norm')
def convert_group_norm(ctx):

    input = get_arg(ctx, 'input', pos=0, default=None)
    num_groups = get_arg(ctx, 'num_groups', pos=1, default=None)
    weight = get_arg(ctx, 'weight', pos=2, default=None)
    bias = get_arg(ctx, 'bias', pos=3, default=None)
    eps = get_arg(ctx, 'eps', pos=4, default=1e-5)
    output = ctx.method_return


    input_trt, eps_trt = add_missing_trt_tensors(ctx.network, [input, eps])
    
    shape = list(input.shape)
    split_shape = [shape[0]] + [num_groups, shape[1] // num_groups] + shape[2:]
    split_shape = tuple(split_shape)
    keepdim = True

    # split into groups
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims = split_shape
    a = layer.get_output(0)


    # compute mean over groups
    reduce_dims = tuple(range(2, len(split_shape)))
    axes = torch_dim_to_trt_axes(reduce_dims)
    layer = ctx.network.add_reduce(a, trt.ReduceOperation.AVG, axes, keepdim)
    a_mean = layer.get_output(0)

    # compute stdev over groups
    a_diff = ctx.network.add_elementwise(a, a_mean, trt.ElementWiseOperation.SUB).get_output(0)
    a_dist = ctx.network.add_elementwise(a_diff, a_diff, trt.ElementWiseOperation.PROD).get_output(0)
    a_var = ctx.network.add_reduce(a_dist, trt.ReduceOperation.AVG, axes, keepdim).get_output(0)


    a_var, eps_trt = broadcast_trt_tensors(ctx.network, [a_var, eps_trt], len(split_shape))

    a_var_eps = ctx.network.add_elementwise(a_var, eps_trt, trt.ElementWiseOperation.SUM).get_output(0)
    a_std = ctx.network.add_unary(a_var_eps, trt.UnaryOperation.SQRT).get_output(0)

    # divide by stdev
    b = ctx.network.add_elementwise(a_diff, a_std, trt.ElementWiseOperation.DIV).get_output(0)

    # reshape
    layer = ctx.network.add_shuffle(b)
    layer.reshape_dims = shape

    c = layer.get_output(0)

    # handle affine version
    if weight is not None or bias is not None:
        if weight is not None:
            scale = weight.detach().cpu().numpy()
        else:
            scale = np.ones(input.shape[1])

        if bias is not None:
            bias = bias.detach().cpu().numpy()
        else:
            bias = np.zeros(input.shape[1])

        power = np.ones_like(scale)

        layer = ctx.network.add_scale_nd(c, trt.ScaleMode.CHANNEL, bias, scale, power, 1)
        c = layer.get_output(0)

    output._trt = c


@tensorrt_converter('torch.Tensor.contiguous')
@tensorrt_converter('torch.nn.functional.dropout')
@tensorrt_converter('torch.nn.functional.dropout2d')
@tensorrt_converter('torch.nn.functional.dropout3d')
def convert_functional_identity(ctx):
    input = ctx.method_args[0]
    if not hasattr(input, '_trt'):
        return
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    output._trt = input_trt


def _add_scale_1d2d3d(network, x_trt, mode, offset, scale, power):
    ndim = len(x_trt.shape)
    
    y_trt = x_trt
    
    # shape to 2D
    if ndim != 4:
        layer = network.add_shuffle(y_trt)
        layer.reshape_dims = (x_trt.shape[0], x_trt.shape[1], x_trt.shape[2], -1)  # NCH -> NCHW
        y_trt = layer.get_output(0)
        
    y_trt = network.add_scale(y_trt, mode, offset, scale, power).get_output(0)

    # shape to original dimension
    if ndim != 4:    
        layer = network.add_shuffle(layer.get_output(0))
        layer.reshape_dims = tuple(x_trt.shape)
        y_trt = layer.get_output(0)
    
    return y_trt
        

@tensorrt_converter('torch.instance_norm')
@tensorrt_converter('torch.nn.functional.instance_norm')
def convert_instance_norm(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    running_mean = get_arg(ctx, 'running_mean', pos=1, default=None)
    running_var = get_arg(ctx, 'running_var', pos=2, default=None)
    weight = get_arg(ctx, 'weight', pos=3, default=None)
    bias = get_arg(ctx, 'bias', pos=4, default=None)
    use_input_stats = get_arg(ctx, 'use_input_stats', pos=5, default=True)
    momentum = get_arg(ctx, 'momentum', pos=6, default=0.1)
    eps = get_arg(ctx, 'eps', pos=7, default=1e-05)
    output = ctx.method_return
    
    
    # CASE 1 - USING RUNNING STATISTICS
    if not use_input_stats:
        
        # equivalent to batch norm
        scale = 1.0 / np.sqrt(running_var.detach().cpu().numpy() + eps)
        offset = -running_mean.detach().cpu().numpy() * scale
        power = np.ones_like(scale)
        
        if weight is not None:
            scale *= weight.detach().cpu().numpy()
            offset += bias.detach().cpu().numpy()
            
        result_trt = _add_scale_1d2d3d(ctx.network, input._trt, trt.ScaleMode.CHANNEL, offset, scale, power)
    
        output._trt = result_trt
        
    # CASE 2 - USING INPUT STATS
    else:
        
        eps_np = np.array([eps], dtype=np.float32)
        keep_dims = True
        reduce_axes = torch_dim_to_trt_axes(tuple(range(2, len(input.shape))))
        
        # compute mean over spatial
        mean_trt = ctx.network.add_reduce(input._trt, trt.ReduceOperation.AVG, reduce_axes, keep_dims).get_output(0)
        
        # compute variance over spatial (include eps, to reduce layer count)
        delta_trt = ctx.network.add_elementwise(input._trt, mean_trt, trt.ElementWiseOperation.SUB).get_output(0)
        var_trt = ctx.network.add_scale(delta_trt, trt.ScaleMode.UNIFORM, np.zeros_like(eps_np), np.ones_like(eps_np), 2 * np.ones_like(eps_np)).get_output(0)
        var_trt = ctx.network.add_reduce(var_trt, trt.ReduceOperation.AVG, reduce_axes, keep_dims).get_output(0)
        
        # compute sqrt(var + eps)
        var_trt = ctx.network.add_scale(var_trt, trt.ScaleMode.UNIFORM, eps_np, np.ones_like(eps_np), 0.5 * np.ones_like(eps_np)).get_output(0)
        
        # compute final result
        result_trt = ctx.network.add_elementwise(delta_trt, var_trt, trt.ElementWiseOperation.DIV).get_output(0)
        
        # compute affine (if applicable)
        if weight is not None:
            
            weight_np = weight.detach().cpu().numpy()
            bias_np = bias.detach().cpu().numpy()
            
            result_trt = _add_scale_1d2d3d(ctx.network, result_trt, trt.ScaleMode.CHANNEL, bias_np, weight_np, np.ones_like(bias_np))
        
        output._trt = result_trt

                                                  
@tensorrt_converter('torch.nn.functional.interpolate')
@tensorrt_converter('torch.nn.functional.upsample')
def convert_interpolate(ctx):                                     
    #parse args                     
    input = get_arg(ctx, 'input', pos=0, default=None) 
    size = get_arg(ctx, 'size', pos=1, default=None)
    scale_factor=get_arg(ctx, 'scale_factor', pos=2, default=None)
    mode = get_arg(ctx, 'mode', pos=3, default='nearest')
    align_corners = get_arg(ctx, 'align_corners', pos=4, default=None)

    input_dim = input.dim() - 2
    
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    layer = ctx.network.add_resize(input=input_trt)

    shape = size
    if shape != None:
        if isinstance(shape, collections.Sequence):
            shape = [input.size(0), input.size(1)] + list(shape)
            shape = make_size_wrapper(shape)
        else:
            shape = [input.size(0), input.size(1)] + [shape] * input_dim
            shape = make_size_wrapper(shape)

        # layer.shape = shape (old, static shape)
        layer.set_input(1, shape._trt)

    scales = scale_factor
    if scales != None:
        if not isinstance(scales, collections.Sequence):
            scales = [scales] * input_dim
        layer.scales = [1, 1] + list(scales)

    def configure_resize_trt_10(layer):
        if mode.lower() in ["linear", "bilinear", "trilinear"]:
            layer.resize_mode = trt.InterpolationMode.LINEAR
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
        elif mode.lower() == 'nearest':
            layer.resize_mode = trt.InterpolationMode.NEAREST
        elif mode.lower() == "bicubic":
            layer.resize_mode = trt.InterpolationMode.CUBIC
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
        else:
            raise RuntimeError(f"Interpolation with mode={mode} is not supported by torch2trt.")
        
    
    def configure_resize_trt_pre_10(layer):
        if mode.lower() in ["linear", "bilinear", "trilinear"]:
            layer.resize_mode = trt.ResizeMode.LINEAR
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
        elif mode.lower() == 'nearest':
            layer.resize_mode = trt.ResizeMode.NEAREST
        elif mode.lower() == "bicubic":
            layer.resize_mode = trt.ResizeMode.CUBIC
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
        else:
            raise RuntimeError(f"Interpolation with mode={mode} is not supported by torch2trt.")
            
    if trt_version() >= "10.0":
        configure_resize_trt_10(layer)
    else:
        configure_resize_trt_pre_10(layer)

    if align_corners != None:
        if trt_version() > '8.0':
            if align_corners:
                layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ALIGN_CORNERS
        else:
            layer.align_corners = align_corners

    output._trt = layer.get_output(0)


@tensorrt_converter('torch.nn.functional.layer_norm')
def convert_layer_norm(ctx):
    input = get_arg(ctx, 'input', 0, None)
    shape = get_arg(ctx, 'normalized_shape', 1, None)
    weight = get_arg(ctx, 'weight', 2, None)
    bias = get_arg(ctx, 'bias', 3, None)
    eps = get_arg(ctx, 'eps', 4, 1e-05)
    output = ctx.method_return
    
    input_trt, eps_trt = add_missing_trt_tensors(
        ctx.network,
        [input, eps]
    )
    
    input_trt, eps_trt = broadcast_trt_tensors(
        ctx.network, 
        [input_trt, eps_trt],
        len(output.shape)
    )
    
    if weight is not None:
        _, weight_trt = add_missing_trt_tensors(
            ctx.network,
            [input, weight]
        )
        _, weight_trt = broadcast_trt_tensors(
            ctx.network, 
            [input_trt, weight_trt],
            len(output.shape)
        )
    
    if bias is not None:
        _, bias_trt = add_missing_trt_tensors(
            ctx.network,
            [input, bias]
        )
        _, bias_trt = broadcast_trt_tensors(
            ctx.network, 
            [input_trt, bias_trt],
            len(output.shape)
        )
    
    if isinstance(shape, int):
        shape = (shape,)
    dim = tuple([-i - 1 for i in range(len(shape))])
    dim = torch_dim_resolve_negative(dim, len(input.shape))
    axes = torch_dim_to_trt_axes(dim)
    
    ux = ctx.network.add_reduce(input_trt, trt.ReduceOperation.AVG, axes, keep_dims=True).get_output(0)
    numerator = ctx.network.add_elementwise(input_trt, ux, trt.ElementWiseOperation.SUB).get_output(0)
    varx = ctx.network.add_elementwise(numerator, numerator, trt.ElementWiseOperation.PROD).get_output(0)
    varx = ctx.network.add_reduce(varx, trt.ReduceOperation.AVG, axes, keep_dims=True).get_output(0)
    denom = ctx.network.add_elementwise(varx, eps_trt, trt.ElementWiseOperation.SUM).get_output(0)
    denom = ctx.network.add_unary(denom, trt.UnaryOperation.SQRT).get_output(0)
    y = ctx.network.add_elementwise(numerator, denom, trt.ElementWiseOperation.DIV).get_output(0)
    
    if weight is not None:
        y = ctx.network.add_elementwise(y, weight_trt, trt.ElementWiseOperation.PROD).get_output(0)
        
    if bias is not None:
        y = ctx.network.add_elementwise(y, bias_trt, trt.ElementWiseOperation.SUM).get_output(0)
    
    output._trt = y
    

@tensorrt_converter('torch.nn.functional.linear')
def convert_linear(ctx):
    input = ctx.method_args[0]
    weight = get_arg(ctx, 'weight', 1, None)
    bias = get_arg(ctx, 'bias', 2, None)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    if trt_version() < "10.0":
        # reshape to ...xNx1x1
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = tuple([0]*input.ndim) + (1, 1) 

        bias_trt = trt.Weights(torch_dtype_to_trt(weight.dtype))
        if bias is not None:
            bias_trt = bias.detach().cpu().numpy()
            
        # add fully connected
        layer = ctx.network.add_fully_connected(
            input=layer.get_output(0),
            num_outputs=int(weight.shape[0]),
            kernel=weight.detach().cpu().numpy(),
            bias=bias_trt)

        # reshape back to N
        layer = ctx.network.add_shuffle(layer.get_output(0))
        layer.reshape_dims = tuple([0] * output.ndim)
        output._trt = layer.get_output(0)
    else:
        weight = weight.detach().cpu().numpy()

        if bias is not None:
            bias = bias.detach().cpu().numpy()
        else:
            bias = np.zeros((int(weight.shape[0]),), dtype=weight.dtype)

        bias_shape = [1] * (input.ndim - 1) + [int(weight.shape[0])]
        bias = bias.reshape(bias_shape)

        
        if weight.ndim < input.ndim:
            weight = weight[None, ...]
            
        kernel_const = ctx.network.add_constant(tuple(weight.shape), weight)
        bias_const = ctx.network.add_constant(tuple(bias.shape), bias)

        mm = ctx.network.add_matrix_multiply(
            input_trt,
            trt.MatrixOperation.NONE,
            kernel_const.get_output(0),
            trt.MatrixOperation.TRANSPOSE
        )

        bias_add = ctx.network.add_elementwise(
            mm.get_output(0), 
            bias_const.get_output(0), 
            trt.ElementWiseOperation.SUM
        
        )

        output._trt = bias_add.get_output(0)
        



@tensorrt_converter('torch.nn.functional.log_softmax')
def convert_log_softmax(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    layer = ctx.network.add_softmax(input=input_trt)
    layer = ctx.network.add_unary(input=layer.get_output(0),
            op=trt.UnaryOperation.LOG)
    output._trt = layer.get_output(0)


@tensorrt_converter("torch.matmul")
@tensorrt_converter("torch.Tensor.__matmul__")
def convert_matmul(ctx):
    x = ctx.method_args[0]
    y = ctx.method_args[1]
    z = ctx.method_return

    x_trt, y_trt = add_missing_trt_tensors(ctx.network, [x, y])

    layer = ctx.network.add_matrix_multiply(
        x_trt,
        trt.MatrixOperation.NONE,
        y_trt,
        trt.MatrixOperation.NONE
    )

    z._trt = layer.get_output(0)


@tensorrt_converter("torch.nn.functional.max_pool3d")
@tensorrt_converter("torch.max_pool3d")
@tensorrt_converter("torch.nn.functional.max_pool2d")
@tensorrt_converter("torch.max_pool2d")
@tensorrt_converter("torch.nn.functional.max_pool1d")
@tensorrt_converter("torch.max_pool1d")
def convert_max_pool_nd(ctx):
    # parse args
    input = get_arg(ctx, "input", pos=0, default=None)
    kernel_size = get_arg(ctx, "kernel_size", pos=1, default=None)
    stride = get_arg(ctx, "stride", pos=2, default=None)
    padding = get_arg(ctx, "padding", pos=3, default=0)
    dilation = get_arg(ctx, "dilation", pos=4, default=1)
    ceil_mode = get_arg(ctx, "ceil_mode", pos=5, default=False)

    trt_pooling_type = trt.PoolingType.MAX

    # get input trt tensor (or create constant if it doesn't exist)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    output = ctx.method_return

    ndim = int(input.ndim - 2)

    # get kernel size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size,) * ndim

    # get stride
    if not isinstance(stride, tuple):
        stride = (stride,) * ndim

    # get padding
    if not isinstance(padding, tuple):
        padding = (padding,) * ndim

    # Shuffle layer to unsqueeze another dimension for 2D max pooling.
    if ndim == 1:
        kernel_size = kernel_size + (1,)
        stride = stride + (1,)
        padding = padding + (0,)
        unsqueeze_layer = ctx.network.add_shuffle(input_trt)
        set_layer_precision(ctx, unsqueeze_layer)
        unsqueeze_layer.reshape_dims = tuple([0]*input.ndim) + (1,) 
        pool_input = unsqueeze_layer.get_output(0)
    else:
        pool_input = input_trt

    pooling_layer = ctx.network.add_pooling_nd(
        input=pool_input, type=trt_pooling_type, window_size=kernel_size
    )

    pooling_layer.stride_nd = stride
    pooling_layer.padding_nd = padding

    if ceil_mode:
        pooling_layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    
    if ndim == 1:
        squeeze_layer = ctx.network.add_shuffle(pooling_layer.get_output(0))
        set_layer_precision(ctx, squeeze_layer)
        squeeze_layer.reshape_dims = tuple([0] * input.ndim)

        output._trt = squeeze_layer.get_output(0)
    else:
        output._trt = pooling_layer.get_output(0)



@tensorrt_converter("torch.nn.functional.avg_pool3d")
@tensorrt_converter("torch.avg_pool3d")
@tensorrt_converter("torch.nn.functional.avg_pool2d")
@tensorrt_converter("torch.avg_pool2d")
@tensorrt_converter("torch.nn.functional.avg_pool1d")
@tensorrt_converter("torch.avg_pool1d")
def convert_avg_pool_nd(ctx):
    # parse args
    input = get_arg(ctx, 'input', pos=0, default=None)
    kernel_size = get_arg(ctx, 'kernel_size', pos=1, default=None)
    stride = get_arg(ctx, 'stride', pos=2, default=None)
    padding = get_arg(ctx, 'padding', pos=3, default=0)
    ceil_mode = get_arg(ctx, 'ceil_mode', pos=4, default=False)
    count_include_pad = get_arg(ctx, 'count_include_pad', pos=5, default=True)
    
    # get input trt tensor (or create constant if it doesn't exist)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    input_dim = input.dim() - 2

    # get kernel size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * input_dim

    # get stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * input_dim

    # get padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * input_dim

    # Shuffle layer to unsqueeze another dimension for 2D max pooling.
    if input_dim == 1:
        kernel_size = kernel_size + (1,)
        stride = stride + (1,)
        padding = padding + (0,)
        unsqueeze_layer = ctx.network.add_shuffle(input_trt)
        set_layer_precision(ctx, unsqueeze_layer)
        unsqueeze_layer.reshape_dims = tuple([0]*input.ndim) + (1,) 
        pool_input = unsqueeze_layer.get_output(0)
    else:
        pool_input = input_trt

    pooling_layer = ctx.network.add_pooling_nd(
        input=pool_input, type=trt.PoolingType.AVERAGE, window_size=kernel_size)
    
    pooling_layer.stride_nd = stride
    pooling_layer.padding_nd = padding
    pooling_layer.average_count_excludes_padding = not count_include_pad
    
    if ceil_mode:
        pooling_layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    if input_dim == 1:
        squeeze_layer = ctx.network.add_shuffle(pooling_layer.get_output(0))
        set_layer_precision(ctx, squeeze_layer)
        squeeze_layer.reshape_dims = tuple([0] * input.ndim)

        output._trt = squeeze_layer.get_output(0)
    else:
        output._trt = pooling_layer.get_output(0)



def __convert_max_elementwise(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.MAX)
    output._trt = layer.get_output(0)


def __convert_max_reduce(ctx):
    input = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=tuple(range(0, len(input.shape))))
    keepdim = get_arg(ctx, 'keepdim', pos=2, default=False)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    if isinstance(ctx.method_return, torch.Tensor):
        output_val = ctx.method_return
    else:
        output_val = ctx.method_return[0]
        output_idx = ctx.method_return[1]
    layer = ctx.network.add_reduce(input_trt,  trt.ReduceOperation.MAX, torch_dim_to_trt_axes(dim), keepdim)
    output_val._trt = layer.get_output(0)


@tensorrt_converter('torch.max')
@tensorrt_converter('torch.Tensor.max')
def convert_max(ctx):
    if len(ctx.method_args) > 1 and isinstance(ctx.method_args[1], torch.Tensor):
        __convert_max_elementwise(ctx)
    else:
        __convert_max_reduce(ctx)


@tensorrt_converter('torch.mean')
@tensorrt_converter('torch.Tensor.mean')
def convert_mean(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    
    dim = get_arg(ctx, "dim", 1, None)
    
    if dim is None:
        dim = [i for i in range(input.ndim)]

    # convert list to tuple
    if isinstance(dim, list):
        dim = tuple(dim)
        
    if not isinstance(dim, tuple):
        dim = (dim, )
        
    # create axes bitmask for reduce layer
    axes = torch_dim_to_trt_axes(dim)
        
    # get whether to keep dimensions
    if 'keepdim' in ctx.method_kwargs:
        keep_dims = ctx.method_kwargs['keepdim']
    elif len(ctx.method_args) == 3:
        keep_dims = ctx.method_args[2]
    else:
        keep_dims = False
        
    layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.AVG, axes, keep_dims)
    output._trt = layer.get_output(0)


def __convert_min_elementwise(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.MIN)
    output._trt = layer.get_output(0)


def __convert_min_reduce(ctx):
    input = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=tuple(range(0, len(input.shape))))
    keepdim = get_arg(ctx, 'keepdim', pos=2, default=False)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    if isinstance(ctx.method_return, torch.Tensor):
        output_val = ctx.method_return
    else:
        output_val = ctx.method_return[0]
        output_idx = ctx.method_return[1]
    layer = ctx.network.add_reduce(input_trt,  trt.ReduceOperation.MIN, torch_dim_to_trt_axes(dim), keepdim)
    output_val._trt = layer.get_output(0)


@tensorrt_converter('torch.min')
@tensorrt_converter('torch.Tensor.min')
def convert_min(ctx):
    if len(ctx.method_args) > 1 and isinstance(ctx.method_args[1], torch.Tensor):
        __convert_min_elementwise(ctx)
    else:
        __convert_min_reduce(ctx)


@tensorrt_converter('torch.fmod')
def convert_mod(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))
    # we can not use ElementWiseOperation.FLOOR_DIV directly because Torch truncate negative result toward 0
    # but TensorRT FLOOR_DIV op toward -Inf
    # sign = ab / |ab|
    # floordiv result: sign * (|a| // |b|)
    ab_layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.PROD)
    abs_ab_layer = ctx.network.add_unary(ab_layer.get_output(0), trt.UnaryOperation.ABS)
    sign_layer = ctx.network.add_elementwise(ab_layer.get_output(0), abs_ab_layer.get_output(0),
                                             trt.ElementWiseOperation.DIV)
    abs_a_layer = ctx.network.add_unary(input_a_trt, trt.UnaryOperation.ABS)
    abs_b_layer = ctx.network.add_unary(input_b_trt, trt.UnaryOperation.ABS)
    abs_floor_layer = ctx.network.add_elementwise(abs_a_layer.get_output(0), abs_b_layer.get_output(0),
                                                  trt.ElementWiseOperation.FLOOR_DIV)
    # a % b  =  a - (a//b) * b
    floordiv_layer = ctx.network.add_elementwise(sign_layer.get_output(0), abs_floor_layer.get_output(0),
                                            trt.ElementWiseOperation.PROD)
    prod_layer = ctx.network.add_elementwise(floordiv_layer.get_output(0), input_b_trt, trt.ElementWiseOperation.PROD)
    sub_layer = ctx.network.add_elementwise(input_a_trt, prod_layer.get_output(0), trt.ElementWiseOperation.SUB)
    output._trt = sub_layer.get_output(0)


@tensorrt_converter('torch.Tensor.__imod__')
@tensorrt_converter('torch.Tensor.__mod__')
# we need separate converter for operator because for some reason Torch use truncation toward -Inf for this op.
# bug is filed: https://github.com/pytorch/pytorch/issues/52425
# but for now we have to convert model exactly
def convert_mod(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))
    # a % b  =  a - (a//b) * b
    floordiv_layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.FLOOR_DIV)
    prod_layer = ctx.network.add_elementwise(floordiv_layer.get_output(0), input_b_trt, trt.ElementWiseOperation.PROD)
    mod_layer = ctx.network.add_elementwise(input_a_trt, prod_layer.get_output(0), trt.ElementWiseOperation.SUB)
    output._trt = mod_layer.get_output(0)


@tensorrt_converter('torch.mul')
@tensorrt_converter('torch.Tensor.mul_')
@tensorrt_converter('torch.Tensor.__imul__')
@tensorrt_converter('torch.Tensor.__mul__')
@tensorrt_converter('torch.Tensor.__rmul__')
def convert_mul(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.PROD)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.Tensor.narrow')
@tensorrt_converter('torch.narrow')
def convert_narrow(ctx):
    inputs = get_arg(ctx, 'input', pos=0, default=None)  
    start = get_arg(ctx, 'start', pos=2, default=None)
    output = ctx.method_return
    shape = list(inputs.shape)
    start = [0]*len(shape)
    stride = [1]*len(shape)
    dim = ctx.method_args[1] if get_arg(ctx, 'dim', pos=1, default=0) >=0 else len(shape)+get_arg(ctx, 'dim', pos=1, default=0)
    
    start[dim] = ctx.method_args[2]
    shape[dim] = ctx.method_args[3] 
    # not consider batch dimension
    input_trt = trt_(ctx.network,inputs)
    layer = ctx.network.add_slice(input=input_trt,start=start, shape=shape,stride=stride)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.ne')
@tensorrt_converter('torch.Tensor.__ne__')
def convert_ne(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))
    layer_1 = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.EQUAL)
    layer_2 = ctx.network.add_unary(layer_1.get_output(0), trt.UnaryOperation.NOT)
    output._trt = layer_2.get_output(0)


@tensorrt_converter('torch.nn.functional.normalize')
def convert_normalize(ctx):
    # get args
    input = get_arg(ctx, 'input', pos=0, default=None)
    p = get_arg(ctx, 'p', pos=1, default=2)
    dim = get_arg(ctx, 'dim', pos=2, default=1)
    eps = get_arg(ctx, 'eps', pos=3, default=1e-12)
    
#     input_trt = input._trt
    output = ctx.method_return
    
    # add broadcastable scalar constants to network
    input_trt, eps_trt, p_trt, p_inv_trt = add_missing_trt_tensors(ctx.network, [input, eps, p, 1.0 / p])
    input_trt, eps_trt, p_trt, p_inv_trt = broadcast_trt_tensors(ctx.network, [input_trt, eps_trt, p_trt, p_inv_trt], len(input_trt.shape))
    
    # compute norm = sum(abs(x)**p, dim=dim)**(1./p)
    norm = ctx.network.add_unary(input_trt, trt.UnaryOperation.ABS).get_output(0)
    norm = ctx.network.add_elementwise(norm, p_trt, trt.ElementWiseOperation.POW).get_output(0)
    norm = ctx.network.add_reduce(norm, trt.ReduceOperation.SUM, torch_dim_to_trt_axes(dim), keep_dims=True).get_output(0)
    norm = ctx.network.add_elementwise(norm, p_inv_trt, trt.ElementWiseOperation.POW).get_output(0)
    
    # clamp norm = max(norm, eps)
    norm = ctx.network.add_elementwise(norm, eps_trt, trt.ElementWiseOperation.MAX).get_output(0)
    
    # divide input by norm
    output._trt = ctx.network.add_elementwise(input_trt, norm, trt.ElementWiseOperation.DIV).get_output(0)
    

@tensorrt_converter('torch.nn.functional.pad')
def convert_pad(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    
    pad = get_arg(ctx, "pad", 1, None)
    mode = get_arg(ctx, "mode", 2, "constant")
    value = get_arg(ctx, "value", 3, 0.)

    pre_padding = (pad[2], pad[0])
    post_padding = (pad[3], pad[1])
    
    # mode / value are ignored since not supported by TensorRT
    
    layer = ctx.network.add_padding(input_trt, pre_padding, post_padding)
    output._trt = layer.get_output(0)
    

@tensorrt_converter('torch.Tensor.permute')
def convert_permute(ctx):
    input = ctx.method_args[0]

    if not hasattr(input, '_trt'):
        return 
        
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    
    # permutation -1 because TRT does not include batch dim
    if isinstance(ctx.method_args[1], int):
        permutation = tuple(ctx.method_args[1:])  # handle permute(a, b, c)
    else:
        permutation = tuple(ctx.method_args[1])   # handle permute([a, b, c])
        
    # assert(permutation[0] == 0)  # cannot move batch dim
    
    layer = ctx.network.add_shuffle(input_trt)
    layer.second_transpose = tuple(permutation)
   
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.pow')
@tensorrt_converter('torch.Tensor.__ipow__')
@tensorrt_converter('torch.Tensor.__pow__')
def convert_pow(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.POW)
    output._trt = layer.get_output(0)

    
@tensorrt_converter('torch.Tensor.__rpow__')
def convert_rpow(ctx):
    input_a = ctx.method_args[1]
    input_b = ctx.method_args[0]  # flipped for rpow
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.POW)
    output._trt = layer.get_output(0)
    

@tensorrt_converter('torch.nn.functional.prelu')
def convert_prelu(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    weight = get_arg(ctx, 'weight', pos=1, default=None)
    output = ctx.method_return
    
    weight_shape = [1] * (len(input.shape))
    weight_shape[1] = weight.numel()
    
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
   
    # y = prelu(x) = relu(x) - alpha * relu(-x)
    weight_trt = ctx.network.add_constant(weight_shape, -weight.detach().view(weight_shape).cpu().numpy()).get_output(0) # detach so considered leaf
    
    # x >= 0
    a = ctx.network.add_activation(input_trt, trt.ActivationType.RELU).get_output(0)
    
    # x <= 0
    b = ctx.network.add_unary(input_trt, trt.UnaryOperation.NEG).get_output(0)
    b = ctx.network.add_activation(b, trt.ActivationType.RELU).get_output(0)
    b = ctx.network.add_elementwise(b, weight_trt, trt.ElementWiseOperation.PROD).get_output(0)
    
    # y = a + b
    y = ctx.network.add_elementwise(a, b, trt.ElementWiseOperation.SUM)
    
    output._trt = y.get_output(0)


@tensorrt_converter('torch.prod')
@tensorrt_converter('torch.Tensor.prod')
def convert_prod(ctx):
    input = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=tuple(range(1, len(input.shape))))
    keepdim = get_arg(ctx, 'keepdim', pos=2, default=False)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    layer = ctx.network.add_reduce(input_trt,  trt.ReduceOperation.PROD, torch_dim_to_trt_axes(dim), keepdim)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.relu')
@tensorrt_converter('torch.relu_')
@tensorrt_converter('torch.nn.functional.relu')
@tensorrt_converter('torch.nn.functional.relu_')
@tensorrt_converter('torch.Tensor.relu')
def convert_relu(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    layer = ctx.network.add_activation(
        input=input_trt, type=trt.ActivationType.RELU)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.nn.functional.relu6')
def convert_relu6(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return

    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input, 6])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))

    layer = ctx.network.add_activation(
        input=input_a_trt, type=trt.ActivationType.RELU)
    layer = ctx.network.add_elementwise(
        layer.get_output(0), input_b_trt, trt.ElementWiseOperation.MIN)

    output._trt = layer.get_output(0)

    
@tensorrt_converter('torch.roll')
@tensorrt_converter('torch.Tensor.roll')
def convert_roll(ctx):
    input = get_arg(ctx, 'input', 0, None)
    shifts = get_arg(ctx, 'shifts', 1, None)
    dims = get_arg(ctx, 'dims', 2, None)
    output = ctx.method_return
    
    assert dims is not None, "roll converter only supports roll when dims is specified"
    
    ndim = input.ndim
    
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    try:
        iter(shifts)
    except:
        shifts = (shifts,)
        dims = (dims,)
    
    start = [0] * ndim
    shape = tuple([int(d) for d in input.shape])
    stride = [1] * ndim
    
    for s, d in zip(shifts, dims):
        start[d] = (-s) % shape[d]
    
    start = tuple(start)
    shape = tuple(shape)
    stride = tuple(stride)
    
    shape_dynamic = ctx.network.add_shape(input._trt).get_output(0)
    layer = ctx.network.add_slice(
        input_trt,
        start,  # [1:] to exclude batch
        shape,
        stride
    )
    layer.set_input(2, shape_dynamic)
    layer.mode = trt.SliceMode.WRAP
    
    output._trt = layer.get_output(0)
    

@tensorrt_converter('torch.nn.functional.sigmoid')
@tensorrt_converter('torch.sigmoid')
@tensorrt_converter('torch.Tensor.sigmoid')
def convert_sigmoid(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SIGMOID)
    output._trt = layer.get_output(0)
    

@tensorrt_converter('torch.nn.functional.silu')
def convert_silu(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    output = ctx.method_return
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SIGMOID)
    layer = ctx.network.add_elementwise(input_trt, layer.get_output(0), trt.ElementWiseOperation.PROD)
    
    output._trt = layer.get_output(0)
    

@tensorrt_converter('torch.Tensor.softmax')
@tensorrt_converter('torch.nn.functional.softmax')
def convert_softmax(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    # get dims from args or kwargs
    if 'dim' in ctx.method_kwargs:
        dim = ctx.method_kwargs['dim']
    elif len(ctx.method_args) >= 2:
        dim = ctx.method_args[1]
        
    # convert negative dims
    if dim < 0:
        dim = len(input.shape) + dim

    axes = torch_dim_to_trt_axes(dim)

    layer = ctx.network.add_softmax(input=input_trt)
    layer.axes = axes

    output._trt = layer.get_output(0)


@tensorrt_converter('torch.Tensor.squeeze')
@tensorrt_converter('torch.squeeze')
def convert_squeeze(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return
    dim = get_arg(ctx, 'dim', pos=1, default=None)

    if dim is None:
        dim = tuple([i for i in range(input.ndim)])

    dim = torch_dim_resolve_negative(dim, input.ndim) 
    # if dim < 0:
    #     dim = len(input.shape) + dim
    # assert dim >= 0

    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    input_shape_trt = ctx.network.add_shape(input_trt).get_output(0)
    new_shape_trt = []

    # get shape before flatten
    for i in range(input.ndim):
        if input.size(i) == 1 and (dim is None) or (i == dim):
            continue # skip 1 dimensions
        else:
            new_shape_trt.append(
                ctx.network.add_slice(input_shape_trt, [i], [1], [1]).get_output(0)
            )

    new_shape_trt = ctx.network.add_concatenation(new_shape_trt).get_output(0)

    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, new_shape_trt)
    output._trt = layer.get_output(0)


def unsqueeze(ctx, input, dim):
    layer = ctx.network.add_shuffle(trt_(ctx.network, input))

    shape = input.shape[:dim] + (1,) + input.shape[dim:]
    layer.reshape_dims = tuple(shape)

    return layer.get_output(0)


@tensorrt_converter('torch.stack')
def convert_stack(ctx):
    inputs = get_arg(ctx, 'input', pos=0, default=None)
    dim = get_arg(ctx, 'dim', pos=1, default=0)

    # Reverse negative dims.
    if dim < 0:
        dim = len(inputs[0].shape) - abs(dim + 1)

    output = ctx.method_return
    trt_inputs = [unsqueeze(ctx, i, dim) for i in inputs]

    layer = ctx.network.add_concatenation(inputs=trt_inputs)
    layer.axis = dim
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.sub')
@tensorrt_converter('torch.Tensor.__isub__')
@tensorrt_converter('torch.Tensor.__sub__')
def convert_sub(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.SUB)
    output._trt = layer.get_output(0)

    
@tensorrt_converter('torch.Tensor.__rsub__')
def convert_sub(ctx):
    input_a = ctx.method_args[1]
    input_b = ctx.method_args[0]  # flipped for rsub
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape))
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.SUB)
    output._trt = layer.get_output(0)
    

@tensorrt_converter('torch.sum')
@tensorrt_converter('torch.Tensor.sum')
def convert_sum(ctx):
    input = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=tuple(range(1, len(input.shape))))
    keepdim = get_arg(ctx, 'keepdim', pos=2, default=False)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    layer = ctx.network.add_reduce(input_trt,  trt.ReduceOperation.SUM, torch_dim_to_trt_axes(dim), keepdim)
    output._trt = layer.get_output(0)
        

@tensorrt_converter('torch.nn.functional.tanh')
@tensorrt_converter('torch.tanh')
def convert_tanh(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.TANH)
    output._trt = layer.get_output(0)
    

@tensorrt_converter('torch.tensor')
def convert_tensor(ctx):
    output = ctx.method_return
    layer = ctx.network.add_constant(tuple(output.shape), output.detach().cpu().numpy() )
    output._trt = layer.get_output(0)



@tensorrt_converter("torch.Tensor.transpose")
@tensorrt_converter('torch.transpose')
def convert_transpose(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    # permutation -1 because TRT does not include batch dim
    permutation = list(range(len(input.shape)))
    dim0 = torch_dim_resolve_negative(ctx.method_args[1], input.ndim)[0]
    dim1 = torch_dim_resolve_negative(ctx.method_args[2], input.ndim)[0]
    permutation[dim0] = dim1
    permutation[dim1] = dim0
    layer = ctx.network.add_shuffle(input_trt)
    layer.second_transpose = tuple(permutation)
    output._trt = layer.get_output(0)


def __convert_unary(ctx, op):
    input = get_arg(ctx, 'input', pos=0, default=None)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    layer = ctx.network.add_unary(input_trt, op)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.exp')
@tensorrt_converter('torch.exp_')
@tensorrt_converter('torch.Tensor.exp')
@tensorrt_converter('torch.Tensor.exp_')
def convert_exp(ctx):
    __convert_unary(ctx, trt.UnaryOperation.EXP)


@tensorrt_converter('torch.log')
@tensorrt_converter('torch.log_')
@tensorrt_converter('torch.Tensor.log')
@tensorrt_converter('torch.Tensor.log_')
def convert_log(ctx):
    __convert_unary(ctx, trt.UnaryOperation.LOG)


@tensorrt_converter('torch.sqrt')
@tensorrt_converter('torch.sqrt_')
@tensorrt_converter('torch.Tensor.sqrt')
@tensorrt_converter('torch.Tensor.sqrt_')
def convert_sqrt(ctx):
    __convert_unary(ctx, trt.UnaryOperation.SQRT)


@tensorrt_converter('torch.reciprocal')
@tensorrt_converter('torch.reciprocal_')
@tensorrt_converter('torch.Tensor.reciprocal')
@tensorrt_converter('torch.Tensor.reciprocal_')
def convert_reciprocal(ctx):
    __convert_unary(ctx, trt.UnaryOperation.RECIP)


@tensorrt_converter('torch.abs')
@tensorrt_converter('torch.abs_')
@tensorrt_converter('torch.Tensor.abs')
@tensorrt_converter('torch.Tensor.abs_')
def convert_abs(ctx):
    __convert_unary(ctx, trt.UnaryOperation.ABS)


@tensorrt_converter('torch.neg')
@tensorrt_converter('torch.neg_')
@tensorrt_converter('torch.Tensor.neg')
@tensorrt_converter('torch.Tensor.__neg__')
@tensorrt_converter('torch.Tensor.neg_')
def convert_neg(ctx):
    __convert_unary(ctx, trt.UnaryOperation.NEG)


@tensorrt_converter('torch.sin')
@tensorrt_converter('torch.sin_')
@tensorrt_converter('torch.Tensor.sin')
@tensorrt_converter('torch.Tensor.sin_')
def convert_sin(ctx):
    __convert_unary(ctx, trt.UnaryOperation.SIN)


@tensorrt_converter('torch.cos')
@tensorrt_converter('torch.cos_')
@tensorrt_converter('torch.Tensor.cos')
@tensorrt_converter('torch.Tensor.cos_')
def convert_cos(ctx):
    __convert_unary(ctx, trt.UnaryOperation.COS)


@tensorrt_converter('torch.tan')
@tensorrt_converter('torch.tan_')
@tensorrt_converter('torch.Tensor.tan')
@tensorrt_converter('torch.Tensor.tan_')
def convert_cos(ctx):
    __convert_unary(ctx, trt.UnaryOperation.TAN)


@tensorrt_converter('torch.sinh')
@tensorrt_converter('torch.sinh_')
@tensorrt_converter('torch.Tensor.sinh')
@tensorrt_converter('torch.Tensor.sinh_')
def convert_sinh(ctx):
    __convert_unary(ctx, trt.UnaryOperation.SINH)


@tensorrt_converter('torch.cosh')
@tensorrt_converter('torch.cosh_')
@tensorrt_converter('torch.Tensor.cosh')
@tensorrt_converter('torch.Tensor.cosh_')
def convert_cosh(ctx):
    __convert_unary(ctx, trt.UnaryOperation.COSH)


@tensorrt_converter('torch.asin')
@tensorrt_converter('torch.asin_')
@tensorrt_converter('torch.Tensor.asin')
@tensorrt_converter('torch.Tensor.asin_')
def convert_asin(ctx):
    __convert_unary(ctx, trt.UnaryOperation.ASIN)


@tensorrt_converter('torch.acos')
@tensorrt_converter('torch.acos_')
@tensorrt_converter('torch.Tensor.acos')
@tensorrt_converter('torch.Tensor.acos_')
def convert_acos(ctx):
    __convert_unary(ctx, trt.UnaryOperation.ACOS)


@tensorrt_converter('torch.atan')
@tensorrt_converter('torch.atan_')
@tensorrt_converter('torch.Tensor.atan')
@tensorrt_converter('torch.Tensor.atan_')
def convert_atan(ctx):
    __convert_unary(ctx, trt.UnaryOperation.ATAN)


@tensorrt_converter('torch.ceil')
@tensorrt_converter('torch.ceil_')
@tensorrt_converter('torch.Tensor.ceil')
@tensorrt_converter('torch.Tensor.ceil_')
def convert_ceil(ctx):
    __convert_unary(ctx, trt.UnaryOperation.CEIL)


@tensorrt_converter('torch.floor')
@tensorrt_converter('torch.floor_')
@tensorrt_converter('torch.Tensor.floor')
@tensorrt_converter('torch.Tensor.floor_')
def convert_floor(ctx):
    __convert_unary(ctx, trt.UnaryOperation.FLOOR)



@tensorrt_converter('torch.Tensor.unsqueeze')
@tensorrt_converter('torch.unsqueeze')
def convert_unsqueeze(ctx):
    input = ctx.method_args[0]

    if not hasattr(input, '_trt'):
        return

    dim = get_arg(ctx, 'dim', pos=1, default=None)
    assert(dim is not None)
    output = ctx.method_return

    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    input_shape_trt = ctx.network.add_shape(input_trt).get_output(0)
    new_shape_trt = []

    for i in range(input.ndim):
        # copy input dim
        new_shape_trt.append(
            ctx.network.add_slice(input_shape_trt, [i], [1], [1]).get_output(0)
        )
    
    # add unsqueeze dim
    new_shape_trt.insert(
        dim,
        ctx.network.add_constant([1], np.array([1], dtype=trt_int_dtype())).get_output(0)
    )

    new_shape_trt = ctx.network.add_concatenation(new_shape_trt).get_output(0)

    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, new_shape_trt)
    output._trt = layer.get_output(0)



@tensorrt_converter('torch.Tensor.view')
@tensorrt_converter('torch.Tensor.reshape')
def convert_view(ctx):
    input = ctx.method_args[0]
    if not hasattr(input, '_trt'):
        return
    
    try:
        iter(ctx.method_args[1])
        size = make_size_wrapper(ctx.method_args[1])
    except:
        size = make_size_wrapper(ctx.method_args[1:])

    output = ctx.method_return

    layer = ctx.network.add_shuffle(input._trt)
    layer.set_input(1, size._trt)
    output._trt = layer.get_output(0)

