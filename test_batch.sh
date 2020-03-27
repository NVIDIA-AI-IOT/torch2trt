#!/bin/bash

OUTPUT=batch_tests_nano.md


python3 -m torch2trt.test --name=googlenet --include=torch2trt.tests.torchvision.batch_tests --torch2trt_kwargs='{"max_batch_size": 1}' -o $OUTPUT
python3 -m torch2trt.test --name=googlenet --include=torch2trt.tests.torchvision.batch_tests --torch2trt_kwargs='{"max_batch_size": 2}' -o $OUTPUT
python3 -m torch2trt.test --name=googlenet --include=torch2trt.tests.torchvision.batch_tests --torch2trt_kwargs='{"max_batch_size": 4}' -o $OUTPUT
python3 -m torch2trt.test --name=googlenet --include=torch2trt.tests.torchvision.batch_tests --torch2trt_kwargs='{"max_batch_size": 8}' -o $OUTPUT

python3 -m torch2trt.test --name=resnet18 --include=torch2trt.tests.torchvision.batch_tests --torch2trt_kwargs='{"max_batch_size": 1}' -o $OUTPUT
python3 -m torch2trt.test --name=resnet18 --include=torch2trt.tests.torchvision.batch_tests --torch2trt_kwargs='{"max_batch_size": 2}' -o $OUTPUT
python3 -m torch2trt.test --name=resnet18 --include=torch2trt.tests.torchvision.batch_tests --torch2trt_kwargs='{"max_batch_size": 4}' -o $OUTPUT
python3 -m torch2trt.test --name=resnet18 --include=torch2trt.tests.torchvision.batch_tests --torch2trt_kwargs='{"max_batch_size": 8}' -o $OUTPUT

python3 -m torch2trt.test --name=resnet50 --include=torch2trt.tests.torchvision.batch_tests --torch2trt_kwargs='{"max_batch_size": 1}' -o $OUTPUT
python3 -m torch2trt.test --name=resnet50 --include=torch2trt.tests.torchvision.batch_tests --torch2trt_kwargs='{"max_batch_size": 2}' -o $OUTPUT
python3 -m torch2trt.test --name=resnet50 --include=torch2trt.tests.torchvision.batch_tests --torch2trt_kwargs='{"max_batch_size": 4}' -o $OUTPUT
python3 -m torch2trt.test --name=resnet50 --include=torch2trt.tests.torchvision.batch_tests --torch2trt_kwargs='{"max_batch_size": 8}' -o $OUTPUT

python3 -m torch2trt.test --name=densenet121 --include=torch2trt.tests.torchvision.batch_tests --torch2trt_kwargs='{"max_batch_size": 1}' -o $OUTPUT
python3 -m torch2trt.test --name=densenet121 --include=torch2trt.tests.torchvision.batch_tests --torch2trt_kwargs='{"max_batch_size": 2}' -o $OUTPUT
python3 -m torch2trt.test --name=densenet121 --include=torch2trt.tests.torchvision.batch_tests --torch2trt_kwargs='{"max_batch_size": 4}' -o $OUTPUT
python3 -m torch2trt.test --name=densenet121 --include=torch2trt.tests.torchvision.batch_tests --torch2trt_kwargs='{"max_batch_size": 8}' -o $OUTPUT
