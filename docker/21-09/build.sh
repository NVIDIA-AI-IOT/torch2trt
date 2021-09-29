#!/bin/bash

docker build -t torch2trt:21-09 -f $(pwd)/docker/21-09/Dockerfile .