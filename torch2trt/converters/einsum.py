from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


def has_opt_einsum():
    try:
        from opt_einsum import contract
        return True
    except:
        return False
    
    
def einsum_remove_batch(expr):
    expr = expr.replace(' ', '')
    if '->' in expr:
        ins, outs = expr.split('->')
        # assume first dim is batch
        ins = ','.join([x[1:] for x in ins.split(',')])
        outs = ','.join([x[1:] for x in outs.split(',')])

        expr = '->'.join([ins, outs])
    else:
        ins = expr
        ins = ','.join([x[1:] for x in ins.split(',')])
        expr = ins
        
    return expr


@tensorrt_converter('torch.einsum', enabled=has_opt_einsum())
def convert_einsum(ctx):
    
    from opt_einsum import contract
    
    equation = ctx.method_args[0]
    operands = ctx.method_args[1:]
    operands_trt = add_missing_trt_tensors(
        ctx.network,
        operands,
    )
    outputs = ctx.method_return
    equation = einsum_remove_batch(equation)
    outputs_trt = contract(equation, *operands_trt, backend='torch2trt')
    
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
        outputs_trt = (outputs_trt,)
        
    for out, out_trt in zip(outputs, outputs_trt):
        out._trt = out_trt
        
        
class Einsum(torch.nn.Module):
    
    def __init__(self, expr):
        super().__init__()
        self.expr = expr
        
    def forward(self, *args):
        return torch.einsum(self.expr, *args)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4), (1, 4, 5)], enabled=has_opt_einsum())
def test_einsum():
    return Einsum('bij,bjk->bik')