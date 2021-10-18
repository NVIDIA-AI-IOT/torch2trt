#!/bin/bash


docker run --gpus all -it --rm -v $(pwd):/torch2trt torch2trt:21-06 