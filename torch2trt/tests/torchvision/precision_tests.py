import torch
import torchvision
from torch2trt.module_test import add_module_test

    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=False, int8_mode=False)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, int8_mode=False)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=False, int8_mode=True)
def googlenet():
    return torchvision.models.googlenet(pretrained=False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=False, int8_mode=False)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, int8_mode=False)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=False, int8_mode=True)
def resnet18():
    return torchvision.models.resnet18(pretrained=False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=False, int8_mode=False)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, int8_mode=False)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=False, int8_mode=True)
def resnet50():
    return torchvision.models.resnet50(pretrained=False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=False, int8_mode=False)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, int8_mode=False)
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=False, int8_mode=True)
def densenet121():
    return torchvision.models.densenet121(pretrained=False)
