import torch
import torchvision
from torch2trt.module_test import add_module_test


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)['out']
    

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)  
def deeplabv3_resnet50():
    bb = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
    model = ModelWrapper(bb)
    return model


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def deeplabv3_resnet101():
    bb = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False)
    model = ModelWrapper(bb)
    return model


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def fcn_resnet50():
    bb = torchvision.models.segmentation.fcn_resnet50(pretrained=False)
    model = ModelWrapper(bb)
    return model


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def fcn_resnet101():
    bb = torchvision.models.segmentation.fcn_resnet101(pretrained=False)
    model = ModelWrapper(bb)
    return model