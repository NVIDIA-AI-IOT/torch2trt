import tensorrt as trt
from torch2trt.torch2trt import tensorrt_converter, get_arg, torch_dim_resolve_negative, add_missing_trt_tensors, torch_dim_to_trt_axes
from torch2trt.module_test import add_module_test


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
    input_shape_reduce_slice_trt = ctx.network.add_slice(
        input_shape_trt, 
        (start_dim,),
        (1 + end_dim - start_dim,), 
        (1,)
    ).get_output(0)

    reduce_size_trt = ctx.network.add_reduce(input_shape_reduce_slice_trt, trt.ReduceOperation.PROD, 1, True).get_output(0)
    new_shape_trt.append(reduce_size_trt)

    # get shape after flatten
    for i in range(end_dim + 1, input.ndim):
        dim_trt = ctx.network.add_slice(input_shape_trt, [i], [1], [1]).get_output(0)
        new_shape_trt.append(dim_trt)

    new_shape_trt = ctx.network.add_concatenation(new_shape_trt).get_output(0)

    # do flatten with shuffle layer
    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, new_shape_trt)
    output._trt = layer.get_output(0)