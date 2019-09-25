import torch
import torchvision


class ModuleTest(object):
    def __init__(self, module_fn, dtype, device, input_shapes, **torch2trt_kwargs):
        self.module_fn = module_fn
        self.dtype = dtype
        self.device = device
        self.input_shapes = input_shapes
        self.torch2trt_kwargs = torch2trt_kwargs
        
    def module_name(self):
        return self.module_fn.__module__ + '.' + self.module_fn.__name__


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
    ModuleTest(torchvision.models.mobilenet_v2, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True),
]


def add_module_test(dtype, device, input_shapes, **torch2trt_kwargs):
    def register_module_test(module):
        global MODULE_TESTS
        MODULE_TESTS += [ModuleTest(module, dtype, device, input_shapes, **torch2trt_kwargs)]
        return module
    return register_module_test
