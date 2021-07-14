import torch 
import torch.nn as nn
import numpy as np 
import torchvision
import argparse
import os,sys 
import torch.optim as optim 
from datasets.cifar10 import Cifar10Loaders
from models.models import vanilla_cnn
from models.resnet import resnet18 , resnet34
from utils.utilities import calculate_accuracy , add_missing_keys, transfer_learning_resnet18,transfer_learning_resnet34, mapping_names
from parser import parse_args
import time
from torch2trt import torch2trt
import tensorrt as trt 

def main():
    args = parse_args()

    ## Create an output dir
    output_dir_path = args.od + args.en
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        dir_name=output_dir_path 
    else:
        counter=1
        dir_name = output_dir_path
        new_dir_name = dir_name
        while os.path.exists(new_dir_name):
            new_dir_name = dir_name + "_" + str(counter)
            counter +=1 
        os.makedirs(new_dir_name)
        dir_name=new_dir_name

    print("===>> Output folder = {}".format(dir_name))
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)
    
    loaders = Cifar10Loaders()
    train_loader = loaders.train_loader()
    test_loader = loaders.test_loader()

    if args.m =="resnet18":
        if args.netqat:
            model=resnet18(qat_mode=True)
        else:
            model=resnet18()
    elif args.m =="resnet34":
        if args.netqat:
            model=resnet34(qat_mode=True)
        else:
            model=resnet34()
    elif args.m == 'resnet34-tl':
        model = transfer_learning_resnet34()
    elif args.m == "resnet18-tl": ## resnet18 transfer learning
        model=transfer_learning_resnet18()
    else:
        raise NotImplementedError("model {} is not defined".format(args.m))

    if args.cuda:
        model = model.cuda()

    best_test_accuracy=0
    if args.v:
        print("======>>> keys present in state dict at model creation")
        for k,_ in model.state_dict().items():
            print(k)

    if args.load_ckpt:
        model.eval()
        checkpoint = torch.load(args.load_ckpt)
        if args.partial_ckpt:
            model_state = checkpoint['model_state_dict']
            if args.v:
                print("====>>>>> keys present in the ckpt state dict")
                for k,_ in model_state.items():
                    print(k)
            if args.tl:
                model_state = mapping_names(model_state)
            new_state_dict = add_missing_keys(model.state_dict(),model_state)
            model.load_state_dict(new_state_dict,strict=True)
        else:
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9)
    if args.load_ckpt:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("===>>> Checkpoint loaded successfully from {} at epoch {} ".format(args.load_ckpt,epoch))

    print("===>> Training started")
    for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
        running_loss=0.0
        start=time.time()
        model.train()
        for i, data in enumerate(train_loader,0):
            inputs, labels = data

            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss +=loss.item()
        
        if epoch > 0 and  epoch % args.lrdt == 0:
            print("===>> decaying learning rate at epoch {}".format(epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.94

        running_loss /= len(train_loader)
        end = time.time()
        test_accuracy = calculate_accuracy(model,test_loader)

        print("Epoch: {0} | Loss: {1} | Test accuracy: {2}| Time Taken (sec): {3} ".format(epoch+1, np.around(running_loss,6), test_accuracy, np.around((end-start),4)))

        ##Save the best checkpoint
        if test_accuracy > best_test_accuracy:
            best_ckpt_filename = dir_name + "/ckpt_" + str(epoch)
            best_test_accuracy = test_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
                }, best_ckpt_filename)
    print("Training finished")
    
    ## Running metrics
    if args.test_trt:
        if args.m == 'resnet34-tl' or args.m == 'resnet34':
            model = transfer_learning_resnet34(pretrained=False)
        elif args.m == 'resnet18-tl' or args.m == 'resnet18':
            model= transfer_learning_resnet18(pretrained=False)
        else:
            raise NotImplementedError("model {} is not defined".format(args.m))
        
        model=model.cuda().eval()
        checkpoint = torch.load(best_ckpt_filename)
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        
        pytorch_test_accuracy = calculate_accuracy(model,test_loader)
        rand_in = torch.randn([128,3,32,32],dtype=torch.float32).cuda()

        if args.FP16:
            trt_model_fp16 = torch2trt(model,[rand_in],log_level=trt.Logger.INFO,fp16_mode=True,max_batch_size=128)
            trtfp16_test_accuracy = calculate_accuracy(trt_model_fp16,test_loader)
    
        if args.INT8PTC:
            ##preparing calib dataset
            calib_dataset = list()
            for i, sam in enumerate(test_loader):
                calib_dataset.extend(sam[0])
                if i ==5:
                    break

            trt_model_calib_int8 = torch2trt(model,[rand_in],log_level=trt.Logger.INFO,fp16_mode=True,int8_calib_dataset=calib_dataset,int8_mode=True,max_batch_size=128)
            int8_test_accuracy = calculate_accuracy(trt_model_calib_int8,test_loader)

        print("Test Accuracy")
        print("Pytorch model :",pytorch_test_accuracy)
        print("TRT FP16 model :",trtfp16_test_accuracy)
        print("TRT INT8 PTC model :",int8_test_accuracy)


if __name__ == "__main__":
    main()
