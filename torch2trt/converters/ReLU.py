from torch2trt.torch2trt import *


@tensorrt_converter('torch.nn.ReLU.forward')
def convert_ReLU(ctx):
    input = ctx.method_args[1]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    layer = ctx.network.add_activation(
        input=input_trt, type=trt.ActivationType.RELU)
    qat_mode = ctx.qat_mode
    fallback_precision = ctx.fallback_precision
    if qat_mode:
        layer.precision = fallback_precision
        layer.set_output_type(0,fallback_precision)
    output._trt = layer.get_output(0)
