import torch
import torch.nn as nn
import numpy as np
import collections
from pytorch_quantization import tensor_quant
from torch2trt.contrib.qat.layers.quant_conv import QuantConvBN2d,QuantConv2d,IQuantConv2d, IQuantConvBN2d
from torch2trt.contrib.qat.layers.quant_activation import QuantReLU, IQuantReLU
import torchvision.models as models  
import re
import timeit

def transfer_learning_resnet18(pretrained=True):
    resnet18 = models.resnet18(pretrained=pretrained)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 10)
    return resnet18

def transfer_learning_resnet34(pretrained=True):
    resnet34 = models.resnet34(pretrained=pretrained)
    num_ftrs = resnet34.fc.in_features
    resnet34.fc = nn.Linear(num_ftrs,10)
    return resnet34

def mapping_names(state_dict):
    '''
    func to map new names
    '''
    new_list = collections.OrderedDict()
    for k,v in state_dict.items():
        if re.search(r'conv\d.weight',k):
            item = re.sub('weight','qconv.0.weight',k)
            print("replacing {} to {}".format(k,item))
            new_list[item]=v
        elif re.search(r'bn\d.\w+',k):
            m = re.search(r'bn\d.\w+',k).group(0)
            word=m.split(".")[-1]
            num = re.search(r'\d',m).group(0)
            new_name = "conv"+num+".qconv.0.bn."+word
            item = re.sub(r'bn\d.\w+',new_name,k)
            print("replacing {} to {}".format(k,item))
            new_list[item]=v
        elif re.search(r'downsample.0.weight',k):
            item = re.sub('weight','qconv.0.weight',k)
            print("replacing {} to {}".format(k,item))
            new_list[item]=v
        elif re.search(r'downsample.1.\w+',k):
            m = re.search(r'downsample.1.\w+',k).group(0)
            word = m.split(".")[-1]
            new_name = "downsample.0.qconv.0.bn."+word
            item = re.sub(r'downsample.1.\w+',new_name,k)
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
            if re.search(r'folded_weight',k):
                item = re.sub("folded_weight","weight",k)
                tensor_size = model_state[item].size()
                model_state_dict[k] = torch.ones(tensor_size)
                print("adding {} with shape {} to the model state dict".format(k,tensor_size))
            elif re.search(r'folded_bias',k):
                item = re.sub("folded_bias","weight",k)
                tensor_size = model_state[item].size()
                model_state_dict[k] = torch.ones(tensor_size[0])
                print("adding {} with shape {} to the model state dict".format(k,tensor_size[0]))
            else:
                print("adding {} to the model state dict".format(k))
                model_state_dict[k]= torch.tensor(127)

    return model_state_dict


## QAT qrapper for ReLU layer: toggles between training and inference

class qrelu(torch.nn.Module):
    def __init__(self,inplace=False,qat=False,infer=False):
        super().__init__()
        if qat:
            if infer:
                self.relu = IQuantReLU(inplace)
            else:
                self.relu = QuantReLU(inplace)
        else:
            self.relu = nn.ReLU(inplace)

    def forward(self,input):
        return self.relu(input)


'''
Wrapper for conv2d + bn + relu layer. 
Toggles between QAT mode(on and off)
Toggles between QAT training and inference

In QAT mode:
    conv(quantized_weight) + BN + ReLU + quantized op. 

'''

class qconv2d(torch.nn.Module):
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
            eps: float=1e-5,
            momentum: float=0.1,
            freeze_bn = False,
            act: bool= True,
            norm: bool=True,
            qat: bool=False,
            infer: bool=False):
        super().__init__()
        if qat:
            if infer:
                if norm:
                    layer_list = [IQuantConvBN2d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        dilation=dilation,
                        bias=bias,
                        padding_mode=padding_mode)]

                else:
                    layer_list = [IQuantConv2d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        dilation=dilation,
                        bias=bias,
                        padding_mode=padding_mode)]
            else:
                if norm:
                    layer_list=[QuantConvBN2d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        dilation=dilation,
                        bias=bias,
                        padding_mode=padding_mode,
                        quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)]

                else:
                    layer_list = [QuantConv2d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        dilation=dilation,
                        bias=bias,
                        padding_mode=padding_mode,
                        quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)]
           
            if act:
                if infer:
                    layer_list.append(IQuantReLU())
                else:
                    layer_list.append(QuantReLU())
            
            self.qconv = nn.Sequential(*layer_list)
    
        else:
            layer_list=[
                    nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        bias=bias,
                        groups=groups)]
            if norm:
                layer_list.append(nn.BatchNorm2d(out_channels))
           
            if act:
                layer_list.append(nn.ReLU())
            
            self.qconv = nn.Sequential(*layer_list)

    def forward(self,inputs):
        return self.qconv(inputs)


def calculate_accuracy(model,data_loader, is_cuda=True):
    correct=0
    total=0
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

 
