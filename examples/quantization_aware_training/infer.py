import torch 
import torch.nn as nn
import numpy as np 
import argparse
import os,sys 
from resnet import resnet18
from parser import parse_args
from torch2trt import torch2trt
import tensorrt as trt

def main():
    args = parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(78543)

    if args.cuda:
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)
    

    if args.m == "resnet18":
        if args.netqat:
            model=resnet18(qat_mode=True,infer=True)
        else:
            model=resnet18()
    else:
        raise NotImplementedError("{} model not found".format(args.m))


    model = model.cuda().eval()

    print(model)
    for k,v in model.state_dict().items():
        if 'learned_amax' in k:
            print(v)

    rand_in = torch.randn([1,3,32,32],dtype=torch.float32).cuda()
    
    #Converting the model to TRT

    trt_model = torch2trt(model,[rand_in],log_level=trt.Logger.INFO,fp16_mode=True,int8_mode=True,max_batch_size=1,qat_mode=True,strict_type_constraints=True)

if __name__ == "__main__":
    main()

