from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

        
def __convert_unary(ctx, op):
    input = get_arg(ctx, 'input', pos=0, default=None)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    layer = ctx.network.add_unary(input_trt, op)
    output._trt = layer.get_output(0)
    

class UnaryModule(torch.nn.Module):
    def __init__(self, fn):
        super(UnaryModule, self).__init__()
        self.fn = fn
        
    def forward(self, x):
        return self.fn(x)
    
# EXP : Exponentiation


@tensorrt_converter('torch.exp')
@tensorrt_converter('torch.exp_')
@tensorrt_converter('torch.Tensor.exp')
@tensorrt_converter('torch.Tensor.exp_')
def convert_exp(ctx):
    __convert_unary(ctx, trt.UnaryOperation.EXP)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_exp():
    return UnaryModule(lambda x: torch.exp(x))


#  LOG : Log (base e)


@tensorrt_converter('torch.log')
@tensorrt_converter('torch.log_')
@tensorrt_converter('torch.Tensor.log')
@tensorrt_converter('torch.Tensor.log_')
def convert_log(ctx):
    __convert_unary(ctx, trt.UnaryOperation.LOG)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_log():
    return UnaryModule(lambda x: torch.log(x))


# SQRT : Square root


@tensorrt_converter('torch.sqrt')
@tensorrt_converter('torch.sqrt_')
@tensorrt_converter('torch.Tensor.sqrt')
@tensorrt_converter('torch.Tensor.sqrt_')
def convert_sqrt(ctx):
    __convert_unary(ctx, trt.UnaryOperation.SQRT)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_sqrt():
    return UnaryModule(lambda x: torch.sqrt(x))


# RECIP : Reciprocal


@tensorrt_converter('torch.reciprocal')
@tensorrt_converter('torch.reciprocal_')
@tensorrt_converter('torch.Tensor.reciprocal')
@tensorrt_converter('torch.Tensor.reciprocal_')
def convert_reciprocal(ctx):
    __convert_unary(ctx, trt.UnaryOperation.RECIP)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_reciprocal():
    return UnaryModule(lambda x: torch.reciprocal(x))


# ABS : Absolute value


@tensorrt_converter('torch.abs')
@tensorrt_converter('torch.abs_')
@tensorrt_converter('torch.Tensor.abs')
@tensorrt_converter('torch.Tensor.abs_')
def convert_abs(ctx):
    __convert_unary(ctx, trt.UnaryOperation.ABS)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_abs():
    return UnaryModule(lambda x: torch.abs(x))


#  NEG : Negation

@tensorrt_converter('torch.neg')
@tensorrt_converter('torch.neg_')
@tensorrt_converter('torch.Tensor.neg')
@tensorrt_converter('torch.Tensor.__neg__')
@tensorrt_converter('torch.Tensor.neg_')
def convert_neg(ctx):
    __convert_unary(ctx, trt.UnaryOperation.NEG)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_neg():
    return UnaryModule(lambda x: torch.neg(x))


#  SIN : Sine


@tensorrt_converter('torch.sin')
@tensorrt_converter('torch.sin_')
@tensorrt_converter('torch.Tensor.sin')
@tensorrt_converter('torch.Tensor.sin_')
def convert_sin(ctx):
    __convert_unary(ctx, trt.UnaryOperation.SIN)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_sin():
    return UnaryModule(lambda x: torch.sin(x))


#  COS : Cosine


@tensorrt_converter('torch.cos')
@tensorrt_converter('torch.cos_')
@tensorrt_converter('torch.Tensor.cos')
@tensorrt_converter('torch.Tensor.cos_')
def convert_cos(ctx):
    __convert_unary(ctx, trt.UnaryOperation.COS)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_cos():
    return UnaryModule(lambda x: torch.cos(x))


#  |    TAN : Tangent


@tensorrt_converter('torch.tan')
@tensorrt_converter('torch.tan_')
@tensorrt_converter('torch.Tensor.tan')
@tensorrt_converter('torch.Tensor.tan_')
def convert_cos(ctx):
    __convert_unary(ctx, trt.UnaryOperation.TAN)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_tan():
    return UnaryModule(lambda x: torch.tan(x))


#  |    SINH : Hyperbolic sine


@tensorrt_converter('torch.sinh')
@tensorrt_converter('torch.sinh_')
@tensorrt_converter('torch.Tensor.sinh')
@tensorrt_converter('torch.Tensor.sinh_')
def convert_sinh(ctx):
    __convert_unary(ctx, trt.UnaryOperation.SINH)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_sinh():
    return UnaryModule(lambda x: torch.sinh(x))


#  |    COSH : Hyperbolic cosine


@tensorrt_converter('torch.cosh')
@tensorrt_converter('torch.cosh_')
@tensorrt_converter('torch.Tensor.cosh')
@tensorrt_converter('torch.Tensor.cosh_')
def convert_cosh(ctx):
    __convert_unary(ctx, trt.UnaryOperation.COSH)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_cosh():
    return UnaryModule(lambda x: torch.cosh(x))


#  |    ASIN : Inverse sine


@tensorrt_converter('torch.asin')
@tensorrt_converter('torch.asin_')
@tensorrt_converter('torch.Tensor.asin')
@tensorrt_converter('torch.Tensor.asin_')
def convert_asin(ctx):
    __convert_unary(ctx, trt.UnaryOperation.ASIN)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_asin():
    return UnaryModule(lambda x: torch.asin(x))


#  |    ACOS : Inverse cosine


@tensorrt_converter('torch.acos')
@tensorrt_converter('torch.acos_')
@tensorrt_converter('torch.Tensor.acos')
@tensorrt_converter('torch.Tensor.acos_')
def convert_acos(ctx):
    __convert_unary(ctx, trt.UnaryOperation.ACOS)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_acos():
    return UnaryModule(lambda x: torch.acos(x))


#  |    ATAN : Inverse tangent


@tensorrt_converter('torch.atan')
@tensorrt_converter('torch.atan_')
@tensorrt_converter('torch.Tensor.atan')
@tensorrt_converter('torch.Tensor.atan_')
def convert_atan(ctx):
    __convert_unary(ctx, trt.UnaryOperation.ATAN)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_atan():
    return UnaryModule(lambda x: torch.atan(x))


#  |    ASINH : Inverse hyperbolic sine
#  |  
#  |    ACOSH : Inverse hyperbolic cosine
#  |  
#  |    ATANH : Inverse hyperbolic tangent
#  |  

#  CEIL : Ceiling


@tensorrt_converter('torch.ceil')
@tensorrt_converter('torch.ceil_')
@tensorrt_converter('torch.Tensor.ceil')
@tensorrt_converter('torch.Tensor.ceil_')
def convert_ceil(ctx):
    __convert_unary(ctx, trt.UnaryOperation.CEIL)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_ceil():
    return UnaryModule(lambda x: torch.ceil(x))


#  FLOOR : Floor
        

@tensorrt_converter('torch.floor')
@tensorrt_converter('torch.floor_')
@tensorrt_converter('torch.Tensor.floor')
@tensorrt_converter('torch.Tensor.floor_')
def convert_floor(ctx):
    __convert_unary(ctx, trt.UnaryOperation.FLOOR)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_floor():
    return UnaryModule(lambda x: torch.floor(x))