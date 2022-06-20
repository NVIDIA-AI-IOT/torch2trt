from torch2trt import *
from .module_test import ModuleTest, MODULE_TESTS
import time
import argparse
import re
import runpy
import traceback
from termcolor import colored
import math
import tempfile
import numpy as np

def pSNR(model_op,trt_op):
    #model_op = model_op.cpu().detach().numpy().flatten()
    #trt_op = trt_op.cpu().detach().numpy().flatten()

    # Calculating Mean Squared Error
    mse = np.sum(np.square(model_op - trt_op)) / len(model_op)
    # Calcuating peak signal to noise ratio
    try:
    	psnr_db = 20 * math.log10(np.max(abs(model_op))) - 10 * math.log10(mse)
    except:
        psnr_db = np.nan
    return mse,psnr_db



def run(self, serialize=False):
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

    if serialize:
        with tempfile.TemporaryFile() as f:
            torch.save(module_trt.state_dict(), f)
            f.seek(0)
            module_trt = TRTModule()
            module_trt.load_state_dict(torch.load(f))
            
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
    if not isinstance(outputs_trt, tuple):
        outputs_trt = (outputs_trt,)
    
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

	## calculate peak signal to noise ratio
    assert(len(outputs) == len(outputs_trt))
	
    ## Check if output is boolean
    # if yes, then dont calculate psnr
    if outputs[0].dtype == torch.bool:
        mse = np.nan
        psnr_db = np.nan
    else:
        model_op = []
        trt_op = []
        for i in range(len(outputs)):
            model_op.extend(outputs[i].detach().cpu().numpy().flatten())
            trt_op.extend(outputs_trt[i].detach().cpu().numpy().flatten())
        model_op = np.array(model_op)
        trt_op = np.array(trt_op)
        mse,psnr_db = pSNR(model_op,trt_op)
    
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
    
    return max_error,psnr_db,mse, fps, fps_trt, ms, ms_trt
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', help='Test output file path', type=str, default='torch2trt_test.md')
    parser.add_argument('--name', help='Regular expression to filter modules to test by name', type=str, default='.*')
    parser.add_argument('--tolerance', help='Maximum error to print warning for entry', type=float, default='-1')
    parser.add_argument('--include', help='Addition python file to include defining additional tests', action='append', default=[])
    parser.add_argument('--use_onnx', help='Whether to test using ONNX or torch2trt tracing', action='store_true')
    parser.add_argument('--serialize', help='Whether to use serialization / deserialization of TRT modules before test', action='store_true')
    args = parser.parse_args()
    
    for include in args.include:
        runpy.run_module(include)
        
    num_tests, num_success, num_tolerance, num_error, num_tolerance_psnr = 0, 0, 0, 0, 0
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
                
            max_error,psnr_db,mse, fps, fps_trt, ms, ms_trt = run(test, serialize=args.serialize)

            # write entry
            line = '| %70s | %s | %25s | %s | %.2E | %.2f | %.2E | %.3g | %.3g | %.3g | %.3g |' % (name, test.dtype.__repr__().split('.')[-1], str(test.input_shapes), str(test.torch2trt_kwargs), max_error,psnr_db,mse, fps, fps_trt, ms, ms_trt)
        
            if args.tolerance >= 0 and max_error > args.tolerance:
                print(colored(line, 'yellow'))
                num_tolerance += 1
            elif psnr_db < 100:
                print(colored(line, 'magenta'))
                num_tolerance_psnr +=1
            else:
                print(line)
            num_success += 1
        except:
            line = '| %s | %s | %s | %s | N/A | N/A | N/A | N/A | N/A |' % (name, test.dtype.__repr__().split('.')[-1], str(test.input_shapes), str(test.torch2trt_kwargs))
            print(colored(line, 'red'))
            num_error += 1
            tb = traceback.format_exc()
            print(tb)
            
        with open(args.output, 'a+') as f:
            f.write(line + '\n')
    
    print('NUM_TESTS: %d' % num_tests)
    print('NUM_SUCCESSFUL_CONVERSION: %d' % num_success)
    print('NUM_FAILED_CONVERSION: %d' % num_error)
    print('NUM_ABOVE_TOLERANCE: %d' % num_tolerance)
    print('NUM_pSNR_TOLERANCE: %d' %num_tolerance_psnr)
