import torch 
import torch.nn as nn
import numpy as np 
import torchvision
import argparse
import os,sys 
import torch.optim as optim 
from datasets.cifar10 import Cifar10Loaders
from models.models import vanilla_cnn
from models.resnet import resnet18
from utils.utilities import calculate_accuracy , add_missing_keys, transfer_learning_resnet18, mapping_names 
from parser import parse_args
import time
from torch.optim.lr_scheduler import StepLR

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
    elif args.m == "vanilla_cnn":
        if args.netqat:
            model=vanilla_cnn(qat_mode=True)
        else:
            model=vanilla_cnn()
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
    
    print(model)
    for k,v in model.state_dict().items():
        print(k)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9)
    if args.load_ckpt:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("===>>> Checkpoint loaded successfully from {} at epoch {} ".format(args.load_ckpt,epoch))

    #scheduler = StepLR(optimizer, step_size=7)
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
            #scheduler.step()

            running_loss +=loss.item()
        
        if epoch > 0 and  epoch % args.lrdt == 0:
            print("===>> decaying learning rate at epoch {}".format(epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.94

        running_loss /= len(train_loader)
        end = time.time()
        test_accuracy = calculate_accuracy(model,test_loader)

        print("Epoch: {0} | Loss: {1} | Test accuracy: {2}| Time Taken (sec): {3} ".format(epoch+1, np.around(running_loss,6), test_accuracy, np.around((end-start),4)))

        best_ckpt_filename = dir_name + "/ckpt_" + str(epoch)
        ##Save the best checkpoint
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
                }, best_ckpt_filename)
    print("Training finished")

if __name__ == "__main__":
    main()
