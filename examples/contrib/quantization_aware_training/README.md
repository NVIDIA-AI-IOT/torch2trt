## QAT working example

This example is using QAT library open sourced by nvidia. [Github link](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization)

## Directory overview

1. This directory contains
   1. `dataset` : contains code for cifar-10 dataset
   2. `layers` : contains implementation for inference. More details under `layers/README.md`
   3. `models`: contains two models. `resnet18` and `vanilla_cnn`
   4. `utils` : contains various utility functions for loading state dict, custom wrapper for training and inference & calculating accuracy during training
   5. `train.py` and `infer.py` : contains code for training and inference (including trt conversion)

2. Usually, nvidia quantization library doesn't provide control per layer for quantization. Custom wrapper under `utils/utilities.py` helps us in quantization selective layers in our model.


## Workflow

Workflow consists of three parts. 
1. Train without quantization:

Here pretrained weights from imagenet are used. 

`python train.py --m <model> --pretrain --num_epochs <num_epochs> --test_trt --FP16 --INT8PTC`

2. Train with quantization (weights are mapped using a custom function to make sure that each weight is loaded correctly)

`python train.py --m <model> --quantize --partial_ckpt --load_ckpt /tmp/pytorch_exp/{} --num_epochs <num_epochs> --lr 1e-4 --lrdt 10`

3. Infer with and without TRT

`python infer.py --m <model> --load_ckpt /tmp/pytorch_exp_1/ckpt_{} --quantize --INT8QAT`


## Accuracy Results 

| Model | FP32 | FP16 | INT8 (QAT) | INT8(PTC) |
|-------|------|------|------------|----------|
| Resnet18 | 83.78 | 83.77 | 83.78 | 83.78 |
| Resnet34 | 85.13 | 85.11 | 84.99 | 84.95 |  
| Resnet50 | 87.56|87.54 |87.49 |87.38 |

Models were intially trained for 40 epochs and then fine tuned with QAT on for 10 epochs.

**Please note that the idea behind these experiments is to see if TRT conversion is working properly rather than achieving industry standard accuracy results**

## Future Work

- Add results for EfficientNet and Mobilenet
