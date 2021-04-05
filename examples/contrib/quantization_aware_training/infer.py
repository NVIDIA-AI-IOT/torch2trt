import timeit
import torch 
import torch.nn as nn
import numpy as np 
import torchvision
import argparse
import os,sys 
from datasets.cifar10 import Cifar10Loaders
from utils.utilities import calculate_accuracy, timeGraph,printStats
from models.resnet import resnet18,resnet34
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
    elif args.m == "resnet34":
        if args.netqat:
            model=resnet34(qat_mode=True,infer=True)
        else:
            model=resnet34()
    else:
        raise NotImplementedError("{} model not found".format(args.m))


    model = model.cuda().eval()

    if args.load_ckpt:
        checkpoint = torch.load(args.load_ckpt)
        if not args.netqat:
            checkpoint = mapping_names_resnets(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        print("===>>> Checkpoint loaded successfully from {} ".format(args.load_ckpt))
    
    test_accuracy = calculate_accuracy(model,test_loader)
    print(" Test accuracy for Pytorch model: {0} ".format(test_accuracy))
    rand_in = torch.randn([128,3,32,32],dtype=torch.float32).cuda()
    
    #Converting the model to TRT
    if args.FP16:
        trt_model_fp16 = torch2trt(model,[rand_in],log_level=trt.Logger.INFO,fp16_mode=True,max_batch_size=128)
        test_accuracy = calculate_accuracy(trt_model_fp16,test_loader)
        print(" TRT test accuracy at FP16: {0}".format(test_accuracy))
    
    if args.INT8QAT:
        trt_model_int8 = torch2trt(model,[rand_in],log_level=trt.Logger.INFO,fp16_mode=True,int8_mode=True,max_batch_size=128,qat_mode=True)
        test_accuracy = calculate_accuracy(trt_model_int8,test_loader)
        print(" TRT test accuracy at INT8 QAT: {0}".format(test_accuracy))
    
    if args.INT8PTC:
        ##preparing calib dataset
        calib_dataset = list()
        for i, sam in enumerate(test_loader):
            calib_dataset.extend(sam[0])
            if i ==5:
                break

        trt_model_calib_int8 = torch2trt(model,[rand_in],log_level=trt.Logger.INFO,fp16_mode=True,int8_calib_dataset=calib_dataset,int8_mode=True,max_batch_size=128)
        test_accuracy = calculate_accuracy(trt_model_calib_int8,test_loader)
        print(" TRT test accuracy at INT8 PTC: {0}".format(test_accuracy))

if __name__ == "__main__":
    main()
