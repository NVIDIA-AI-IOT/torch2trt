from torch2trt import *
import torchvision


class ModuleTest(object):
    def __init__(self, module_fn, type, device, input_shapes, max_error=1e-2, **torch2trt_kwargs):
        self.module_fn = module_fn
        self.type = type
        self.device = device
        self.input_shapes = input_shapes
        self.max_error = max_error
        self.torch2trt_kwargs = torch2trt_kwargs
        
    def run(self):
        # create module
        module = self.module_fn()
        module = module.to(self.device)
        module = module.type(self.type)
        module = module.eval()
        
        # create inputs
        inputs = ()
        for shape in self.input_shapes:
            inputs += (torch.ones(shape).to(self.device).type(self.type), )

        # convert module
        module_trt = torch2trt(module, inputs, **self.torch2trt_kwargs)

        # test output against original
        outputs = module(*inputs)
        outputs_trt = module_trt(*inputs)

        if not isinstance(outputs, tuple):
            outputs = (outputs, )

        for i in range(len(outputs)):
            max_error = torch.max(torch.abs(outputs[i] - outputs_trt[i]))
            if max_error > self.max_error:
                raise RuntimeError('Output %d max error exceeded threshold of %f' % (i, self.max_error))
            
            
                
TESTS = {
    'resnet18_fp16': ModuleTest(
        torchvision.models.resnet18,
        torch.float16,
        torch.device('cuda'),
        [(1, 3, 224, 224)],
        max_error=1e-2,
        fp16_mode=True
    ),
}


if __name__ == '__main__':
    for name, test in TESTS.items():
        print('Testing %s ...' % name, end=" ")
        test.run()
        print('PASSED')
        