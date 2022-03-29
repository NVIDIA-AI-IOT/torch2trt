#!/bin/bash

docker build -t torch2trt:21-08 -f $(pwd)/docker/21-08/Dockerfile .
