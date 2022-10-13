#!/bin/bash

VERSION=l4t-35.1.0

docker build -t torch2trt:$VERSION -f $(pwd)/docker/$VERSION/Dockerfile $(pwd)/docker/$VERSION