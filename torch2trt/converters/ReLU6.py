from torch2trt.torch2trt import *


@tensorrt_converter('torch.nn.ReLU6.forward')
def convert_ReLU6(ctx):
    input = ctx.method_args[1]
    output = ctx.method_return

    layer = ctx.network.add_activation(
        input=input._trt, type=trt.ActivationType.RELU)
    shape = (1, ) * len(input._trt.shape)  # broadcast all dimensions
    tensor = 6.0 * torch.ones(shape, dtype=torch_dtype_from_trt(input._trt.dtype)).cpu().numpy()
    trt_6 = ctx.network.add_constant(shape, tensor)
    layer = ctx.network.add_elementwise(
        layer.get_output(0), trt_6.get_output(0), trt.ElementWiseOperation.MIN)

    output._trt = layer.get_output(0)