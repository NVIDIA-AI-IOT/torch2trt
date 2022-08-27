#!/bin/bash


docker run --gpus all -it -d --rm -v $(pwd):/torch2trt torch2trt:21-09