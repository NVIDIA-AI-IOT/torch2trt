#!/bin/bash

OUTPUT=precision_tests.md


python3 -m torch2trt.test --name=googlenet --include=torch2trt.tests.torchvision.precision_tests --torch2trt_kwargs='{"fp16_mode": false, "int8_mode": false}' -o $OUTPUT
python3 -m torch2trt.test --name=googlenet --include=torch2trt.tests.torchvision.precision_tests --torch2trt_kwargs='{"fp16_mode": true, "int8_mode": false}' -o $OUTPUT
python3 -m torch2trt.test --name=googlenet --include=torch2trt.tests.torchvision.precision_tests --torch2trt_kwargs='{"fp16_mode": false, "int8_mode": true}' -o $OUTPUT

python3 -m torch2trt.test --name=resnet18 --include=torch2trt.tests.torchvision.precision_tests --torch2trt_kwargs='{"fp16_mode": false, "int8_mode": false}' -o $OUTPUT
python3 -m torch2trt.test --name=resnet18 --include=torch2trt.tests.torchvision.precision_tests --torch2trt_kwargs='{"fp16_mode": true, "int8_mode": false}' -o $OUTPUT
python3 -m torch2trt.test --name=resnet18 --include=torch2trt.tests.torchvision.precision_tests --torch2trt_kwargs='{"fp16_mode": false, "int8_mode": true}' -o $OUTPUT

python3 -m torch2trt.test --name=resnet50 --include=torch2trt.tests.torchvision.precision_tests --torch2trt_kwargs='{"fp16_mode": false, "int8_mode": false}' -o $OUTPUT
python3 -m torch2trt.test --name=resnet50 --include=torch2trt.tests.torchvision.precision_tests --torch2trt_kwargs='{"fp16_mode": true, "int8_mode": false}' -o $OUTPUT
python3 -m torch2trt.test --name=resnet50 --include=torch2trt.tests.torchvision.precision_tests --torch2trt_kwargs='{"fp16_mode": false, "int8_mode": true}' -o $OUTPUT

python3 -m torch2trt.test --name=densenet121 --include=torch2trt.tests.torchvision.precision_tests --torch2trt_kwargs='{"fp16_mode": false, "int8_mode": false}' -o $OUTPUT
python3 -m torch2trt.test --name=densenet121 --include=torch2trt.tests.torchvision.precision_tests --torch2trt_kwargs='{"fp16_mode": true, "int8_mode": false}' -o $OUTPUT
python3 -m torch2trt.test --name=densenet121 --include=torch2trt.tests.torchvision.precision_tests --torch2trt_kwargs='{"fp16_mode": false, "int8_mode": true}' -o $OUTPUT

