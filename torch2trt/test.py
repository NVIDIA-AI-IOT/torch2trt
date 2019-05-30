from torch2trt import *
import torchvision
import time
import argparse
import re


class ModuleTest(object):
    def __init__(self, module_fn, dtype, device, input_shapes, **torch2trt_kwargs):
        self.module_fn = module_fn
        self.dtype = dtype
        self.device = device
        self.input_shapes = input_shapes
        self.torch2trt_kwargs = torch2trt_kwargs
        
    def module_name(self):
        return self.module_fn.__module__ + '.' + self.module_fn.__name__
    
    def run(self):
        # create module
        module = self.module_fn()
        module = module.to(self.device)
        module = module.type(self.dtype)
        module = module.eval()
        
        # create inputs
        inputs = ()
        for shape in self.input_shapes:
            inputs += (torch.ones(shape).to(self.device).type(self.dtype), )

        # convert module
        module_trt = torch2trt(module, inputs, **self.torch2trt_kwargs)

        # test output against original
        outputs = module(*inputs)
        outputs_trt = module_trt(*inputs)

        if not isinstance(outputs, tuple):
            outputs = (outputs, )

        # compute max error
        max_error = 0
        for i in range(len(outputs)):
            max_error_i = torch.max(torch.abs(outputs[i] - outputs_trt[i]))
            if max_error_i > max_error:
                max_error = max_error_i
        
        # benchmark pytorch
        t0 = time.time()
        for i in range(50):
            outputs = module(*inputs)
        t1 = time.time()
        
        fps = 50.0 / (t1 - t0)
        
        # benchmark tensorrt
        t0 = time.time()
        for i in range(50):
            outputs = module_trt(*inputs)
        t1 = time.time()
        
        fps_trt = 50.0 / (t1 - t0)
        
        return max_error, fps, fps_trt
            
        
MODULE_TESTS = [
    ModuleTest(torchvision.models.alexnet, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.squeezenet1_0, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.squeezenet1_1, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.resnet18, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.resnet34, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.resnet50, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.resnet101, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.resnet152, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.densenet121, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.densenet169, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.densenet201, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.densenet161, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.vgg11, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.vgg13, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.vgg16, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.vgg19, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.vgg11_bn, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.vgg13_bn, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.vgg16_bn, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
    ModuleTest(torchvision.models.vgg19_bn, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
]


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', help='Test output file path', type=str, default='torch2trt_test.md')
    parser.add_argument('--name', help='Regular expression to filter modules to test by name', type=str, default='.*')
    args = parser.parse_args()
    
    # write header
    line0 = '| Name | Data Type | Input Shapes | torch2trt kwargs | Max Error | FPS (PyTorch) | FPS (TensorRT) |'
    line1 = '|------|-----------|--------------|------------------|-----------|---------------|----------------|'
    print(line0)
    print(line1)
    with open(args.output, 'a+') as f:
        f.write(line0 + '\n')
        f.write(line1 + '\n')
        
    for test in MODULE_TESTS:
        
        # filter by module name
        name = test.module_name()
        if not re.search(args.name, name):
            continue
            
        # run test
        max_error, fps, fps_trt = test.run()
        
        # write entry
        line = '| %s | %s | %s | %s | %.2E | %.3g | %.3g |' % (name, test.dtype.__repr__().split('.')[-1], str(test.input_shapes), str(test.torch2trt_kwargs), max_error, fps, fps_trt)
        print(line)
        with open(args.output, 'a+') as f:
            f.write(line + '\n')