import torch
from .torch2trt import *
import unittest
import time
import logging


def benchmark_throughput_fps(module, inputs, num_iter=50):
    torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(num_iter):
        outputs = module(*inputs)
    torch.cuda.current_stream().synchronize()
    t1 = time.time()
    
    return float(num_iter) / (t1 - t0)
    
    
def benchmark_latency_ms(module, inputs, num_iter=50):
    torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(num_iter):
        outputs = module(*inputs)
        torch.cuda.current_stream().synchronize()
    t1 = time.time()
    
    return 1000.0 * (t1 - t0) / float(num_iter)


class ModuleTest(object):
    
    def module(self):
        raise NotImplementedError
    
    def inputs(self):
        raise NotImplementedError
        
    def max_error(self):
        return None
    
    def min_throughput(self):
        return None
    
    def max_latency(self):
        return None
    
    def num_iter(self):
        return 50
        
    def torch2trt_kwargs(self):
        return {}
    
    def test_module(self):
        
        # create module
        module = self.module()
        
        # create inputs
        inputs = self.inputs()
        dtype = inputs[0].dtype
        shapes = [tuple(i.shape) for i in inputs]
        
        # create copy of inputs to handle inplace ops
        inputs_trt = tuple([tensor.clone() for tensor in inputs])
        
        # convert module
        module_trt = torch2trt(module, inputs, **self.torch2trt_kwargs())
        
        # test output against original
        outputs = module(*inputs)
        outputs_trt = module_trt(*inputs_trt)
        
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
            
        # compute max error
        max_error = 0
        for i in range(len(outputs)):
            max_error_i = torch.max(torch.abs(outputs[i] - outputs_trt[i]))
            if max_error_i > max_error:
                max_error
        
        # benchmark throughput
        fps = benchmark_throughput_fps(module, inputs, self.num_iter())
        fps_trt = benchmark_throughput_fps(module_trt, inputs, self.num_iter())
        ms = benchmark_latency_ms(module, inputs, self.num_iter())
        ms_trt = benchmark_latency_ms(module_trt, inputs, self.num_iter())

        logger = logging.getLogger('torch2trt.module_test')
        logger.debug('| %s | %s | %s | %s | %.2E | %.3g | %.3g | %.3g | %.3g |' % (
            self.__class__.__name__,
            dtype,
            shapes,
            self.torch2trt_kwargs(),
            max_error,
            fps,
            fps_trt,
            ms,
            ms_trt
        ))
        
        if self.max_error():
            assert(max_error < self.max_error())
        if self.min_throughput():
            assert(fps_trt > self.min_throughput())
        if self.max_latency():
            assert(ms_trt < self.max_latency())

        
class LegacyModuleTest(object):
    def __init__(self, module_fn, dtype, device, input_shapes, **torch2trt_kwargs):
        self.module_fn = module_fn
        self.dtype = dtype
        self.device = device
        self.input_shapes = input_shapes
        self.torch2trt_kwargs = torch2trt_kwargs
        
    def module_name(self):
        return self.module_fn.__module__ + '.' + self.module_fn.__name__


MODULE_TESTS = [
]


def add_module_test(dtype, device, input_shapes, **torch2trt_kwargs):
    def register_module_test(module):
        global MODULE_TESTS
        MODULE_TESTS += [LegacyModuleTest(module, dtype, device, input_shapes, **torch2trt_kwargs)]
        return module
    return register_module_test