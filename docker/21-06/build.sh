#!/bin/bash

docker build -t torch2trt:21-06 -f $(pwd)/docker/21-06/Dockerfile .