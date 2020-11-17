import torch 
import torch.nn as nn
import numpy as np 
import torchvision
import argparse
import os,sys 
from torch2trt.examples.datasets.cifar10 import Cifar10Loaders
from torch2trt.examples.models.models import vanilla_cnn
from torch2trt.examples.utils.utilities import calculate_accuracy 
from torch2trt.examples.models.resnet import resnet18
from torch2trt.examples.parser import parse_args
from torch2trt import torch2trt
import tensorrt as trt

def main():
    args = parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(78543)

    if args.cuda:
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)
    
    loaders = Cifar10Loaders()
    train_loader = loaders.train_loader()
    test_loader = loaders.test_loader()

    if args.m == "resnet18":
        if args.netqat:
            model=resnet18(qat_mode=True,infer=True)
        else:
            model=resnet18()
    elif args.m == "vanilla_cnn":
        if args.netqat:
            model=vanilla_cnn(qat_mode=True,infer=True)
        else:
            model=vanilla_cnn()
    else:
        raise NotImplementedError("{} model not found".format(args.m))


    model = model.cuda()
    model = model.eval()
    for k, _ in model.state_dict().items():
        print(k)

    if args.load_ckpt:
        checkpoint = torch.load(args.load_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        print("===>>> Checkpoint loaded successfully from {} ".format(args.load_ckpt))
    
    print(model)


    test_accuracy = calculate_accuracy(model,test_loader)
    print(" Test accuracy: {0} ".format(test_accuracy))
    rand_in = torch.randn([32,3,32,32],dtype=torch.float32).cuda()
    
    #Converting the model to TRT

    trt_model = torch2trt(model,[rand_in],log_level=trt.Logger.INFO,fp16_mode=True,int8_mode=True,max_batch_size=32,qat_mode=True,strict_type_constraints=True)
    test_accuracy = calculate_accuracy(trt_model,test_loader)
    print(" TRT test accuracy: {0}".format(test_accuracy))

if __name__ == "__main__":
    main()
