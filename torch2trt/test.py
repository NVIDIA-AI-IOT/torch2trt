from torch2trt import *
from .module_test import ModuleTest, MODULE_TESTS
import time
import argparse
import re
import runpy
from termcolor import colored


def run(self):
    # create module
    module = self.module_fn()
    module = module.to(self.device)
    module = module.type(self.dtype)
    module = module.eval()
    
    # create inputs for conversion
    inputs_conversion = ()
    for shape in self.input_shapes:
        inputs_conversion += (torch.zeros(shape).to(self.device).type(self.dtype), )
        
    # convert module
    module_trt = torch2trt(module, inputs_conversion, max_workspace_size=1 << 20,  **self.torch2trt_kwargs)

    # create inputs for torch/trt.. copy of inputs to handle inplace ops
    inputs = ()
    for shape in self.input_shapes:
        inputs += (torch.randn(shape).to(self.device).type(self.dtype), )
    inputs_trt = tuple([tensor.clone() for tensor in inputs])


    # test output against original
    outputs = module(*inputs)
    outputs_trt = module_trt(*inputs_trt)

    if not isinstance(outputs, tuple):
        outputs = (outputs, )
    
    # compute max error
    max_error = 0
    for i in range(len(outputs)):
        max_error_i = 0
        if outputs[i].dtype == torch.bool:
            max_error_i = torch.sum(outputs[i] ^ outputs_trt[i])
        else:
            max_error_i = torch.max(torch.abs(outputs[i] - outputs_trt[i]))

        if max_error_i > max_error:
            max_error = max_error_i
    
    # benchmark pytorch throughput
    torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(50):
        outputs = module(*inputs)
    torch.cuda.current_stream().synchronize()
    t1 = time.time()
    
    fps = 50.0 / (t1 - t0)
    
    # benchmark tensorrt throughput
    torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(50):
        outputs = module_trt(*inputs)
    torch.cuda.current_stream().synchronize()
    t1 = time.time()
    
    fps_trt = 50.0 / (t1 - t0)
    
    # benchmark pytorch latency
    torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(50):
        outputs = module(*inputs)
        torch.cuda.current_stream().synchronize()
    t1 = time.time()
    
    ms = 1000.0 * (t1 - t0) / 50.0
    
    # benchmark tensorrt latency
    torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(50):
        outputs = module_trt(*inputs)
        torch.cuda.current_stream().synchronize()
    t1 = time.time()
    
    ms_trt = 1000.0 * (t1 - t0) / 50.0
    
    return max_error, fps, fps_trt, ms, ms_trt
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', help='Test output file path', type=str, default='torch2trt_test.md')
    parser.add_argument('--name', help='Regular expression to filter modules to test by name', type=str, default='.*')
    parser.add_argument('--tolerance', help='Maximum error to print warning for entry', type=float, default='-1')
    parser.add_argument('--include', help='Addition python file to include defining additional tests', action='append', default=[])
    args = parser.parse_args()
    
    for include in args.include:
        runpy.run_module(include)
        
    for test in MODULE_TESTS:
        
        # filter by module name
        name = test.module_name()
        if not re.search(args.name, name):
            continue
            
        # run test
        max_error, fps, fps_trt, ms, ms_trt = run(test)
        
        # write entry
        line = '| %s | %s | %s | %s | %.2E | %.3g | %.3g | %.3g | %.3g |' % (name, test.dtype.__repr__().split('.')[-1], str(test.input_shapes), str(test.torch2trt_kwargs), max_error, fps, fps_trt, ms, ms_trt)

        if args.tolerance >= 0 and max_error > args.tolerance:
            print(colored(line, 'yellow'))
        else:
            print(line)

        with open(args.output, 'a+') as f:
            f.write(line + '\n')
