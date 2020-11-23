
import timeit
import torch 
import torch.nn as nn
import numpy as np 
import torchvision
import argparse
import os,sys 
from datasets.cifar10 import Cifar10Loaders
from models.models import vanilla_cnn,vanilla_cnn2
from utils.utilities import calculate_accuracy, timeGraph,printStats 
from models.resnet import resnet18
from parser import parse_args
from torch2trt import torch2trt
import tensorrt as trt
torch.set_printoptions(precision=5)
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
    elif args.m == "vanilla_cnn2":
        if args.netqat:
            model =vanilla_cnn2(qat_mode=True,infer=True)
        else:
            model=vanilla_cnn2()
    elif args.m == "vanilla_cnn":
        if args.netqat:
            model=vanilla_cnn(qat_mode=True,infer=True)
        else:
            model=vanilla_cnn()
    else:
        raise NotImplementedError("{} model not found".format(args.m))


    model = model.cuda().eval()

    if args.load_ckpt:
        checkpoint = torch.load(args.load_ckpt)
        for k,v in checkpoint['model_state_dict'].items():
            if 'learned_amax' in k:
                print(k,v)
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        print("===>>> Checkpoint loaded successfully from {} ".format(args.load_ckpt))
    
    print(model)
    for k,v in model.state_dict().items():
        if 'learned_amax' in k:
            print(k,v)

    test_accuracy = calculate_accuracy(model,test_loader)
    print(" Test accuracy: {0} ".format(test_accuracy))
    rand_in = torch.randn([128,3,32,32],dtype=torch.float32).cuda()
    
    #Converting the model to TRT

    trt_model_int8 = torch2trt(model,[rand_in],log_level=trt.Logger.INFO,fp16_mode=True,int8_mode=True,max_batch_size=128,qat_mode=True,strict_type_constraints=True)
    test_accuracy = calculate_accuracy(trt_model_int8,test_loader)
    print(" TRT test accuracy: {0}".format(test_accuracy))
    timings = timeGraph(trt_model_int8, rand_in, args.iter)
    printStats('int8 trt model', timings, args.b)

    trt_model_fp16 = torch2trt(model,[rand_in],log_level=trt.Logger.INFO,fp16_mode=True,int8_mode=False,max_batch_size=128,qat_mode=False,strict_type_constraints=True)
    test_accuracy = calculate_accuracy(trt_model_fp16,test_loader)
    print(" TRT test accuracy: {0}".format(test_accuracy))
    timings = timeGraph(trt_model_fp16, rand_in, args.iter)
    printStats('fp16 trt model', timings, args.b)



    
    
if __name__ == "__main__":
    main()
