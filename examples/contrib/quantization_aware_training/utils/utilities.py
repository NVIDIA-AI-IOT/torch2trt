import torch
import torch.nn as nn
import numpy as np
import collections
from pytorch_quantization import tensor_quant
from torch2trt.contrib.qat.layers.quant_conv import QuantConv2d
from torch2trt.contrib.qat.layers.quant_pooling import QuantMaxPool2d, QuantAdaptiveAvgPool2d
import torchvision.models as models  
import re
import timeit

def transfer_learning(model : str, pretrained : bool = True):
    if model == "resnet18":
        resnet18 = models.resnet18(pretrained=pretrained)
        num_ftrs = resnet18.fc.in_features
        resnet18.fc = nn.Linear(num_ftrs, 10)
        return resnet18
    elif model =="resnet34":
        resnet34 = models.resnet34(pretrained=pretrained)
        num_ftrs = resnet34.fc.in_features
        resnet34.fc = nn.Linear(num_ftrs,10)
        return resnet34
    elif model == "resnet50":
        resnet50 = models.resnet50(pretrained=pretrained)
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Linear(num_ftrs,10)
        return resnet50
    else:
        return None

def mapping_names(state_dict):
    '''
    func to map new names
    '''
    new_list = collections.OrderedDict()
    for k,v in state_dict.items():
        if re.search(r'conv\d.weight',k):
            item = re.sub('weight','quant.weight',k)
            print("replacing {} to {}".format(k,item))
            new_list[item]=v
        elif re.search(r'downsample.0.weight',k):
            item = re.sub('weight','quant.weight',k)
            print("replacing {} to {}".format(k,item))
            new_list[item]=v
        else:
            print("adding {} to the new list".format(k))
            new_list[k]=v 
    return new_list


def add_missing_keys(model_state,model_state_dict):
    """
    add missing keys and defaulting the values to 1 for _amax counter
    """
    for k,v in model_state.items():
        if k not in model_state_dict.keys():
            print("adding {} to the model state dict".format(k))
            model_state_dict[k]= torch.tensor(127)

    return model_state_dict


'''
Wrapper for conv2d
Toggles between QAT mode(on and off)
'''

class QConv2d(torch.nn.Module):
    """
    common layer for qat and non qat mode
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int=3,
            stride: int=1,
            padding: int=0,
            groups: int=1,
            dilation: int=1,
            bias = None,
            padding_mode: str='zeros',
            qat: bool=False,
            quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL,
            quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR):
        super().__init__()
        if qat:
            self.quant = QuantConv2d(in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=bias,
                padding_mode=padding_mode,
                quant_desc_weight=quant_desc_weight,
                quant_desc_input=quant_desc_input)
        else:
            self.quant = nn.Conv2d(in_channels, 
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    padding_mode=padding_mode)

    def forward(self,inputs):
        return self.quant(inputs)

'''
Wrapper for MaxPool2d
'''

class QMaxPool2d(torch.nn.Module):
    '''
    wrapper for maxpool2d layer to toggle between qat and non-qat mode
    '''
    def __init__(
            self,
            kernel_size,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
            qat_mode = False,
            quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR):
        super().__init__()
        if qat_mode:
            self.quant = QuantMaxPool2d(
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    return_indices=return_indices,
                    ceil_mode=ceil_mode,
                    quant_desc_input=quant_desc_input)
        else:
            self.quant= nn.MaxPool2d(kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    return_indices=return_indices,
                    ceil_mode=ceil_mode)

    def forward(self,input):
        return self.quant(input)
                    
'''
Wrapper for AdaptiveAvgPool2d
'''

class QAdaptiveAvgPool2d(torch.nn.Module):
    '''
    wrapper for AdaptiveAvgPool2d layer to toggle between qat and non-qat mode
    '''
    def __init__(
            self,
            output_size,
            qat_mode = False,
            quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR):
        super().__init__()
        if qat_mode:
            self.quant = QuantAdaptiveAvgPool2d(
                    output_size,
                    quant_desc_input=quant_desc_input)
        else:
            self.quant= nn.AdaptiveAvgPool2d(output_size)

    def forward(self,input):
        return self.quant(input)
 

def calculate_accuracy(model,data_loader, is_cuda=True):
    correct=0
    total=0
    count=0
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images , labels = data
            if is_cuda:
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
            if is_cuda:
                torch.cuda.synchronize()
    acc = correct * 100 / total
    return acc 


def timeGraph(model, input_t, num_loops):
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(20):
            features = model(input_t)

    torch.cuda.synchronize()

    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(num_loops):
            start_time = timeit.default_timer()
            features = model(input_t)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            timings.append(end_time - start_time)
            #print("Iteration {}: {:.6f} s".format(i, end_time - start_time))
    print("Input shape:", input_t.size())
    print("Output features size:", features.size())
    return timings

def printStats(graphName, timings, batch_size):
    times = np.array(timings)
    steps = len(times)
    speeds = batch_size / times
    time_mean = np.mean(times)
    time_med = np.median(times)
    time_99th = np.percentile(times, 99)
    time_std = np.std(times, ddof=0)
    speed_mean = np.mean(speeds)
    speed_med = np.median(speeds)

    msg = ("\n%s =================================\n"
            "batch size=%d, num iterations=%d\n"
            "  Median FPS: %.1f, mean: %.1f\n"
            "  Median latency: %.6f, mean: %.6f, 99th_p: %.6f, std_dev: %.6f\n"
            ) % (graphName,
                batch_size, steps,
                speed_med, speed_mean,
                time_med, time_mean, time_99th, time_std)
    print(msg)

 
