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

# I have an open PR for a change in nvidia library. In the meantime, please use the branch from my fork to run .

RUN git clone https://github.com/SrivastavaKshitij/TensorRT-1.git /sw/TensorRT/ && \
    cd /sw/TensorRT/ && \
    git checkout store_learned_amax_state_dict && \
    git sparse-checkout init --cone && \
    git sparse-checkout set /tools/pytorch-quantization/ && \
    cd tools/pytorch-quantization/ && \
    python setup.py install 

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git /sw/TensorRT/ && \
    cd /sw/TensorRT/ && \
    git fetch origin pull/442/head:PR442 && \
    git checkout PR442 && \
    python setup.py install --plugins

## I will update this command, once I generate the PR for torch2trt
#RUN git clone --recursive https://github.com/SrivastavaKshitij/torch2trt.git /sw/torch2trt && \
#    cd /sw/torch2trt && \
#    git checkout nvidia_quantization && \
#    python setup.py install --plugins

```

Docker build: `docker build -f pytorch_ngc_container_20.09 -t pytorch_ngc_container_20.09 .`

`docker_image=pytorch_ngc_container_20.09`

Docker run : `docker run -e NVIDIA_VISIBLE_DEVICES=0 --gpus 0 -it --shm-size=1g --ulimit memlock=-1  --rm  -v $PWD:/workspace/work $docker_image` 

**Note** : Sparse checkout helps us in checking out a part of the github repo. 

## Workflow

Workflow consists of three parts. 
1. Train without quantization:

`python train.py --m resnet18/vanilla_cnn --num_epochs 30`

2. Train with quantization

`python train.py --m resnet18/vanilla_cnn --netqat --partial_ckpt --load_ckpt /tmp/pytorch_exp/ckpt_{}`

3. Infer with and without TRT

`python infer.py --m resnet18/vanilla_cnn --load_ckpt /tmp/pytorch_exp_1/ckpt_{} --netqat`

