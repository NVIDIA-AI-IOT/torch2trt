import torch
import torchvision
from torch2trt.module_test import add_module_test

    
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, max_batch_size=1)
@add_module_test(torch.float16, torch.device('cuda'), [(2, 3, 224, 224)], fp16_mode=True, max_batch_size=2)
@add_module_test(torch.float16, torch.device('cuda'), [(4, 3, 224, 224)], fp16_mode=True, max_batch_size=4)
@add_module_test(torch.float16, torch.device('cuda'), [(8, 3, 224, 224)], fp16_mode=True, max_batch_size=8)
def googlenet():
    return torchvision.models.googlenet(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, max_batch_size=1)
@add_module_test(torch.float16, torch.device('cuda'), [(2, 3, 224, 224)], fp16_mode=True, max_batch_size=2)
@add_module_test(torch.float16, torch.device('cuda'), [(4, 3, 224, 224)], fp16_mode=True, max_batch_size=4)
@add_module_test(torch.float16, torch.device('cuda'), [(8, 3, 224, 224)], fp16_mode=True, max_batch_size=8)
def resnet18():
    return torchvision.models.resnet18(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, max_batch_size=1)
@add_module_test(torch.float16, torch.device('cuda'), [(2, 3, 224, 224)], fp16_mode=True, max_batch_size=2)
@add_module_test(torch.float16, torch.device('cuda'), [(4, 3, 224, 224)], fp16_mode=True, max_batch_size=4)
@add_module_test(torch.float16, torch.device('cuda'), [(8, 3, 224, 224)], fp16_mode=True, max_batch_size=8)
def resnet50():
    return torchvision.models.resnet50(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True, max_batch_size=1)
@add_module_test(torch.float16, torch.device('cuda'), [(2, 3, 224, 224)], fp16_mode=True, max_batch_size=2)
@add_module_test(torch.float16, torch.device('cuda'), [(4, 3, 224, 224)], fp16_mode=True, max_batch_size=4)
@add_module_test(torch.float16, torch.device('cuda'), [(8, 3, 224, 224)], fp16_mode=True, max_batch_size=8)
def densenet121():
    return torchvision.models.densenet121(pretrained=False)
