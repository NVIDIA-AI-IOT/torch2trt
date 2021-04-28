#!/bin/bash

# generates model cache for use with multiprocessing benchmark
BATCH_SIZE=1

python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu --batch_size=$BATCH_SIZE --use_cache --dla_core=0 # resnet18_int8_gpu_trt_bs1.pth
python3 scripts/benchmark_dla.py --model=resnet18_int8_dla --batch_size=$BATCH_SIZE --use_cache --dla_core=0 # resnet18_int8_dla_trt_bs1.pth
python3 scripts/benchmark_dla.py --model=resnet18_int8_dla --batch_size=$BATCH_SIZE --use_cache --dla_core=1 # resnet18_int8_dla_trt_bs1.pth

python3 scripts/benchmark_dla.py --model=resnet18_fp16_gpu --batch_size=$BATCH_SIZE --use_cache --dla_core=0 # resnet18_fp16_gpu_trt_bs1.pth
python3 scripts/benchmark_dla.py --model=resnet18_fp16_dla --batch_size=$BATCH_SIZE --use_cache --dla_core=0 # resnet18_fp16_dla_trt_bs1.pth
python3 scripts/benchmark_dla.py --model=resnet18_fp16_dla --batch_size=$BATCH_SIZE --use_cache --dla_core=1 # resnet18_fp16_dla_trt_bs1.pth