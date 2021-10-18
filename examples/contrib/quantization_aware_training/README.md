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

## Environment

**Filename** : pytorch_ngc_container_20.09     

```
FROM nvcr.io/nvidia/pytorch:20.09-py3
RUN apt-get update && apt-get install -y software-properties-common && apt-get update
RUN add-apt-repository ppa:git-core/ppa && \
    apt install -y git    

RUN pip install termcolor graphviz

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git /sw/torch2trt/ && \
    cd /sw/torch2trt/scripts && \
	bash build_contrib.sh

```

Docker build: `docker build -f pytorch_ngc_container_20.09 -t pytorch_ngc_container_20.09 .`

`docker_image=pytorch_ngc_container_20.09`

Docker run : `docker run -e NVIDIA_VISIBLE_DEVICES=0 --gpus 0 -it --shm-size=1g --ulimit memlock=-1  --rm  -v $PWD:/workspace/work $docker_image` 

**Important Notes** : 

- Sparse checkout helps us in checking out a part of the github repo. 
- Patch file can be found under `examples/quantization_aware_training/utils`

## Workflow

Workflow consists of three parts. 
1. Train without quantization:

Here pretrained weights from imagenet are used. 

`python train.py --m resnet34-tl / resnet18-tl --num_epochs 45 --test_trt --FP16 --INT8PTC`

2. Train with quantization (weights are mapped using a custom function to make sure that each weight is loaded correctly)

`python train.py --m resnet34/ resnet18 --netqat --partial_ckpt --tl --load_ckpt /tmp/pytorch_exp/{} --num_epochs 25 --lr 1e-4 --lrdt 10`

3. Infer with and without TRT

`python infer.py --m resnet34/resnet18 --load_ckpt /tmp/pytorch_exp_1/ckpt_{} --netqat --INT8QAT`


## Accuracy Results 

| Model | FP32 | FP16 | INT8 (QAT) | INT(PTC) |
|-------|------|------|------------|----------|
| Resnet18 | 83.08 | 83.12 | 83.12 | 83.06 |
| Resnet34 | 84.65 | 84.65 | 83.26 | 84.5 |  


**Please note that the idea behind these experiments is to see if TRT conversion is working properly rather than achieving industry standard accuracy results**

## Future Work

- Add results for Resnet50, EfficientNet and Mobilenet
