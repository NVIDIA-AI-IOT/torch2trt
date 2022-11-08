#!/bin/bash

VERSION=l4t-35.1.0


docker run \
    --network host \
    --ipc host \
    --gpus all \
    -it \
    -d \
    --rm \
    --name=torch2trt \
    -v $(pwd):/torch2trt \
    torch2trt:$VERSION