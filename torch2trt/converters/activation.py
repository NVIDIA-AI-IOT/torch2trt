from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from .unary import UnaryModule


# |    RELU : Rectified Linear activation (impl in relu.py)
#  |    SIGMOID : Sigmoid activation  (impl in sigmoid.py)
#  |    TANH : Hyperbolic Tangent activation  (impl in tanh.py)


#  |    LEAKY_RELU : Leaky Relu activation: f(x) = x if x >= 0, f(x) = alpha * x if x < 0


@tensorrt_converter('torch.nn.functional.leaky_relu')
@tensorrt_converter('torch.nn.functional.leaky_relu_')
def convert_leaky_relu(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    negative_slope = get_arg(ctx, 'negative_slope', pos=1, default=0.01)
    output = ctx.method_return
    
    input_trt = trt_(ctx.network, input)
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.LEAKY_RELU)
    layer.alpha = negative_slope
    
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_leaky_relu():
    return UnaryModule(lambda x: torch.nn.functional.leaky_relu(x))


#  |    ELU : Elu activation: f(x) = x if x >= 0, f(x) = alpha * (exp(x) - 1) if x < 0


@tensorrt_converter('torch.nn.functional.elu')
@tensorrt_converter('torch.nn.functional.elu_')
def convert_elu(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    alpha = get_arg(ctx, 'alpha', pos=1, default=1.0)
    output = ctx.method_return
    
    input_trt = trt_(ctx.network, input)
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.ELU)
    layer.alpha = alpha
    
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_elu():
    return UnaryModule(lambda x: torch.nn.functional.elu(x))


#  |    SELU : Selu activation: f(x) = beta * x if x > 0, f(x) = beta * (alpha * exp(x) - alpha) if x <= 0

@tensorrt_converter('torch.selu')
@tensorrt_converter('torch.selu_')
@tensorrt_converter('torch.nn.functional.selu')
@tensorrt_converter('torch.nn.functional.selu_')
def convert_selu(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    alpha = get_arg(ctx, 'alpha', pos=1, default=1.0)
    output = ctx.method_return
    
    input_trt = trt_(ctx.network, input)
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SELU)
    layer.alpha = 1.6732632423543772848170429916717
    layer.beta = 1.0507009873554804934193349852946
    
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_selu():
    return UnaryModule(lambda x: torch.nn.functional.selu(x))


#  |    SOFTSIGN : Softsign activation: f(x) = x / (1 + \|x\|)


@tensorrt_converter('torch.nn.functional.softsign')
def convert_softsign(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    output = ctx.method_return
    
    input_trt = trt_(ctx.network, input)
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SOFTSIGN)
    
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_softsign():
    return UnaryModule(lambda x: torch.nn.functional.softsign(x))


#  |    SOFTPLUS : Softplus activation: f(x) = alpha * log(exp(beta * x) + 1)


@tensorrt_converter('torch.nn.functional.softplus')
def convert_softplus(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    output = ctx.method_return
    
    input_trt = trt_(ctx.network, input)
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SOFTPLUS)
    
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_softplus():
    return UnaryModule(lambda x: torch.nn.functional.softplus(x))


#  |    CLIP : Clip activation: f(x) = max(alpha, min(beta, x))  (impl in clamp.py)

#  |    HARD_SIGMOID : Hard sigmoid activation: f(x) = max(0, min(1, alpha * x + beta)) (not sure if there is this in Pytorch?)
#  |    SCALED_TANH : Scaled Tanh activation: f(x) = alpha * tanh(beta * x) (not sure if there is this in Pytorch?)
#  |    THRESHOLDED_RELU : Thresholded Relu activation: f(x) = x if x > alpha, f(x) = 0 if x <= alpha (not sure if there is this in Pytorch?)