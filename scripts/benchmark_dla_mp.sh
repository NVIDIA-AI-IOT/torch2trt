#!/bin/bash

# generates model cache for use with multiprocessing benchmark
BATCH_SIZE=1

# INT8

# 3 models GPU 0 models DLA
echo "Profiling 3GPU-0DLA"
python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu --batch_size=$BATCH_SIZE --use_cache --distributed --rank=0 --world_size=3 &
python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu --batch_size=$BATCH_SIZE --use_cache --distributed --rank=1 --world_size=3 &
python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu --batch_size=$BATCH_SIZE --use_cache --distributed --rank=2 --world_size=3 

# 2 models GPU 1 model DLA
echo "Profiling 2GPU-1DLA"
python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu --batch_size=$BATCH_SIZE --use_cache --distributed --rank=0 --world_size=3 &
python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu --batch_size=$BATCH_SIZE --use_cache --distributed --rank=1 --world_size=3 &
python3 scripts/benchmark_dla.py --model=resnet18_int8_dla --batch_size=$BATCH_SIZE --use_cache --dla_core=0 --distributed --rank=2 --world_size=3 

# 1 models GPU 2 model DLA
echo "Profiling 1GPU-2DLA"
python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu --batch_size=$BATCH_SIZE --use_cache --distributed --rank=0 --world_size=3 &
python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu --batch_size=$BATCH_SIZE --use_cache --dla_core=0 --distributed --rank=1 --world_size=3 &
python3 scripts/benchmark_dla.py --model=resnet18_int8_dla --batch_size=$BATCH_SIZE --use_cache --dla_core=1 --distributed --rank=2 --world_size=3 

# python3 scripts/benchmark_dla.py --model=resnet18_int8_dla --batch_size=$BATCH_SIZE --use_cache --dla_core=0 # resnet18_int8_dla_trt_bs1.pth
# python3 scripts/benchmark_dla.py --model=resnet18_int8_dla --batch_size=$BATCH_SIZE --use_cache --dla_core=1 # resnet18_int8_dla_trt_bs1.pth

# python3 scripts/benchmark_dla.py --model=resnet18_fp16_gpu --batch_size=$BATCH_SIZE --use_cache --dla_core=0 # resnet18_fp16_gpu_trt_bs1.pth
# python3 scripts/benchmark_dla.py --model=resnet18_fp16_dla --batch_size=$BATCH_SIZE --use_cache --dla_core=0 # resnet18_fp16_dla_trt_bs1.pth
# python3 scripts/benchmark_dla.py --model=resnet18_fp16_dla --batch_size=$BATCH_SIZE --use_cache --dla_core=1 # resnet18_fp16_dla_trt_bs1.pth