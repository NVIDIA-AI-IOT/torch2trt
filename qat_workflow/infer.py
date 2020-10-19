import torch 
import torch.nn as nn
import numpy as np 
import torchvision
import argparse
import os,sys 
from cifar10 import Cifar10Loaders
from models import cnn
from utilities import calculate_accuracy 
import torch.quantization as quantization
from torch2trt import torch2trt
import tensorrt as trt

def main():

    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(12345)
    batch_size=1    
    loaders = Cifar10Loaders(batch_size=batch_size)
    test_loader = loaders.test_loader()

    model=cnn(qat_mode=True)
    model = model.cuda().eval()
    
    checkpoint = torch.load('checkpoint/ckpt_6')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    print("===>>> Checkpoint loaded successfully from ")

    print(model)
    test_accuracy = calculate_accuracy(model,test_loader)
    print(" Test accuracy: {0} ".format(test_accuracy))
    
    rand_in = torch.randn([batch_size,3,32,32],dtype=torch.float32).cuda()
    trt_model = torch2trt(model,[rand_in],log_level=trt.Logger.INFO,fp16_mode=True,int8_mode=True,max_batch_size=batch_size,QAT_mode=True,strict_type_constraints=True)
    test_accuracy = calculate_accuracy(trt_model,test_loader)
    print(" TRT test accuracy: {0}".format(test_accuracy))

if __name__ == "__main__":
    main()
