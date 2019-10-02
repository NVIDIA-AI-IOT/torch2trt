import torch
import torchvision
from torch2trt.module_test import add_module_test

    
@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def alexnet():
    return torchvision.models.alexnet(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def squeezenet1_0():
    return torchvision.models.squeezenet1_0(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def squeezenet1_1():
    return torchvision.models.squeezenet1_1(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def resnet18():
    return torchvision.models.resnet18(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def resnet34():
    return torchvision.models.resnet34(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def resnet50():
    return torchvision.models.resnet50(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def resnet101():
    return torchvision.models.resnet101(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def resnet152():
    return torchvision.models.resnet152(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def densenet121():
    return torchvision.models.densenet121(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def densenet169():
    return torchvision.models.densenet169(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def densenet201():
    return torchvision.models.densenet201(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def densenet161():
    return torchvision.models.densenet161(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def vgg11():
    return torchvision.models.vgg11(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def vgg13():
    return torchvision.models.vgg13(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def vgg16():
    return torchvision.models.vgg16(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def vgg19():
    return torchvision.models.vgg19(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def vgg11_bn():
    return torchvision.models.vgg11_bn(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def vgg13_bn():
    return torchvision.models.vgg13_bn(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def vgg16_bn():
    return torchvision.models.vgg16_bn(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def vgg19_bn():
    return torchvision.models.vgg19_bn(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def mobilenet_v2():
    return torchvision.models.mobilenet_v2(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def shufflenet_v2_x0_5():
    return torchvision.models.shufflenet_v2_x0_5(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def shufflenet_v2_x1_0():
    return torchvision.models.shufflenet_v2_x1_0(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def shufflenet_v2_x1_5():
    return torchvision.models.shufflenet_v2_x1_5(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def shufflenet_v2_x2_0():
    return torchvision.models.shufflenet_v2_x2_0(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def mnasnet0_5():
    return torchvision.models.mnasnet0_5(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def mnasnet0_75():
    return torchvision.models.mnasnet0_75(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def mnasnet1_0():
    return torchvision.models.mnasnet1_0(pretrained=False)


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def mnasnet1_3():
    return torchvision.models.mnasnet1_3(pretrained=False)