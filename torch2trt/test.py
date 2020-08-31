from torch2trt import *
from .module_test import ModuleTest, MODULE_TESTS
import time
import argparse
import re
import runpy
from termcolor import colored


def GiB(val):
    return val * 1 << 30

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
    with torch2trt(module, inputs_conversion, log_level=trt.Logger.INFO, max_workspace_size= GiB(1), **self.torch2trt_kwargs) as module_trt:

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

        NUM_N = 50
        NUM_F = NUM_N * 1.0
    
        # benchmark pytorch
        torch.cuda.current_stream().synchronize()
        t0 = time.time()
        for i in range(NUM_N):
            outputs = module(*inputs)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
    
        fps = NUM_F / (t1 - t0)
        ms = 1000.0 * (t1 - t0) / NUM_F
    
        # benchmark tensorrt
        torch.cuda.current_stream().synchronize()
        t0 = time.time()
        for i in range(NUM_N):
            outputs = module_trt(*inputs)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
    
        fps_trt = NUM_F / (t1 - t0)
        ms_trt = 1000.0 * (t1 - t0) / NUM_F
    
        return max_error, fps, fps_trt, ms, ms_trt
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', help='Test output file path', type=str, default='torch2trt_test.md')
    parser.add_argument('--name', help='Regular expression to filter modules to test by name', type=str, default='.*')
    parser.add_argument('--tolerance', help='Maximum error to print warning for entry', type=float, default='-1')
    parser.add_argument('--include', help='Addition python file to include defining additional tests', action='append', default=[])
    parser.add_argument('--use_onnx', help='Whether to test using ONNX or torch2trt tracing', action='store_true')
    args = parser.parse_args()
    
    for include in args.include:
        runpy.run_module(include)
        
    num_tests, num_success, num_tolerance, num_error = 0, 0, 0, 0
    for test in MODULE_TESTS:
        
        # filter by module name
        name = test.module_name()
        if not re.search(args.name, name):
            continue
            
        num_tests += 1
        # run test
        try:
            if args.use_onnx:
                test.torch2trt_kwargs.update({'use_onnx': True})
                
            max_error, fps, fps_trt, ms, ms_trt = run(test)

            # write entry
            line = '| %s | %s | %s | %s | %.2E | %.3g | %.3g | %.3g | %.3g | %.3g |' % (name, test.dtype.__repr__().split('.')[-1], str(test.input_shapes), str(test.torch2trt_kwargs), max_error, fps, fps_trt, ms, ms_trt, ms / ms_trt)
        
            if args.tolerance >= 0 and max_error > args.tolerance:
                print(colored(line, 'yellow'))
                num_tolerance += 1
            else:
                print(line)
            num_success += 1
        except:
            line = '| %s | %s | %s | %s | N/A | N/A | N/A | N/A | N/A | N/A |' % (name, test.dtype.__repr__().split('.')[-1], str(test.input_shapes), str(test.torch2trt_kwargs))
            print(colored(line, 'red'))
            num_error += 1
            
        with open(args.output, 'a+') as f:
            f.write(line + '\n')
    
    print('NUM_TESTS: %d' % num_tests)
    print('NUM_SUCCESSFUL_CONVERSION: %d' % num_success)
    print('NUM_FAILED_CONVERSION: %d' % num_error)
    print('NUM_ABOVE_TOLERANCE: %d' % num_tolerance)