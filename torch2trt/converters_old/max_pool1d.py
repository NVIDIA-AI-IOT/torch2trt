from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.max_pool1d')
def convert_max_pool1d(ctx):
    # At the time of this implementation, TensorRT 8.x does not yet support max pooling in 1D using `add_pooling_nd(...)`.
    # As such, we use a workaround here, by unsqueezing another dimension into the input (thus transforming it from
    # (N, C, L) to (N, C, L, 1)) so that we can use 2D max pooling across the last three dimensions.

    input = get_arg(ctx, 'input', pos=0, default=None)
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    kernel_size = get_arg(ctx, 'kernel_size', pos=1, default=None)
    stride = get_arg(ctx, 'stride', pos=2, default=None)
    padding = get_arg(ctx, 'padding', pos=3, default=0)
    dilation = get_arg(ctx, 'dilation', pos=4, default=1)  # Unused.
    return_indices = get_arg(ctx, 'return_indices', pos=5, default=False) # Unused.
    ceil_mode = get_arg(ctx, 'ceil_mode', pos=6, default=False)

    # Convert inputs to be 2d compatible as inputs will always be 1d.
    kernel_size = (kernel_size, 1)
    stride = kernel_size if not stride else (stride, 1)
    padding = (padding, 0)

    # Shuffle layer to unsqueeze another dimension for 2D max pooling.
    unsqueeze_layer = ctx.network.add_shuffle(input_trt)
    set_layer_precision(ctx, unsqueeze_layer)
    unsqueeze_layer.reshape_dims = tuple([0]*input.ndim) + (1,) 
    unsqueeze_trt = unsqueeze_layer.get_output(0)

    # Use 2D max pooling here to fake 1D max pooling.
    pooling_layer = ctx.network.add_pooling_nd(
        input=unsqueeze_trt, type=trt.PoolingType.MAX, window_size=kernel_size
    )
    set_layer_precision(ctx, pooling_layer)
    pooling_layer.stride = stride
    pooling_layer.padding = padding

    if ceil_mode:
        pooling_layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    pooling_trt = pooling_layer.get_output(0)

    # Shuffle layer to squeeze out dimension that was just added for 2D max pooling so return is still in 1D.
    squeeze_layer = ctx.network.add_shuffle(pooling_trt)
    set_layer_precision(ctx, squeeze_layer)
    squeeze_layer.reshape_dims = tuple([0] * input.ndim)
    output._trt = squeeze_layer.get_output(0)


class MaxPool1D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return torch.nn.functional.max_pool1d(x, self.kernel_size, stride=self.stride, padding=self.padding, ceil_mode=self.ceil_mode)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 32)])
def test_max_pool1d_basic():
    return MaxPool1D(2)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 32)], fp16_mode=True)
def test_max_pool1d_fp16_mode():
    return MaxPool1D(2)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 32)], int8_mode=True)
def test_max_pool1d_int8_mode():
    return MaxPool1D(2)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 32)])
def test_max_pool1d_stride():
    return MaxPool1D(2, stride=3)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 32)])
def test_max_pool1d_max_padding():
    return MaxPool1D(2, padding=1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 32)])
def test_max_pool1d_ceil_mode():
    return MaxPool1D(2, ceil_mode=True)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 32)])
@add_module_test(torch.float32, torch.device('cuda'), [(2, 3, 32)], max_batch_size=2)
def test_max_pool1d_all():
    return MaxPool1D(4, stride=3, padding=2, ceil_mode=True)

