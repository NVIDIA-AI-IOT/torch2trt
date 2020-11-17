import torch
import torch.nn as nn
import numpy as np
import collections
from pytorch_quantization import tensor_quant
from torch2trt.qat_layers.quant_conv import QuantConv2d,IQuantConv2d
from torch2trt.qat_layers.quant_linear import QuantLinear,IQuantLinear
from torch2trt.qat_layers.quant_activation import QuantReLU, IQuantReLU

def add_missing_keys(model_state,model_state_dict):
    """
    add missing keys and defaulting the values to 1 for _amax counter
    """
    for k,v in model_state.items():
        if k not in model_state_dict.keys():
            print("adding {} to the model state dict".format(k))
            model_state_dict[k]= torch.tensor(1)

    return model_state_dict
    
## QAT wrapper for linear layer : toggles between inference vs training
class qlinear(torch.nn.Module):
    def __init__(self,in_features,out_features,bias=True,qat=False,infer=False):
        super().__init__()
        if qat:
            if infer:
                self.linear = quantlinear(in_features,out_features,bias)
            else:
                self.linear=QuantLinear(in_features=in_features,out_features=out_features,bias=bias,quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)
        else:
            self.linear=torch.nn.Linear(in_features=in_features,out_features=out_features,bias=bias)

    def forward(self,inputs):
        return self.linear(inputs)

## QAT wrapper for conv + bn + relu layer. as per nvidia qat library only the input of conv and weights are quantized.

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
                layer_list = [quantconv2d(in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    dilation=dilation,
                    bias=bias,
                    padding_mode=padding_mode)]
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
            if norm:
                layer_list.append(nn.BatchNorm2d(out_channels))
           
            if act:
                layer_list.append(nn.ReLU())
            
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


            
