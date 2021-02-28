#!/bin/bash

# generates model cache for use with multiprocessing benchmark
BATCH_SIZE=1

# INT8

# 2 - 0
python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu --batch_size=$BATCH_SIZE --use_cache --distributed --rank=0 --world_size=2 &
python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu --batch_size=$BATCH_SIZE --use_cache --distributed --rank=1 --world_size=2

# 1 - 1
python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu --batch_size=$BATCH_SIZE --use_cache --distributed --rank=0 --world_size=2 &
python3 scripts/benchmark_dla.py --model=resnet18_int8_dla --batch_size=$BATCH_SIZE --use_cache --dla_core=0 --distributed --rank=1 --world_size=2 

# 0 - 2
# echo "Profiling 1GPU-2DLA"
python3 scripts/benchmark_dla.py --model=resnet18_int8_dla --batch_size=$BATCH_SIZE --use_cache --dla_core=0 --distributed --rank=0 --world_size=2 &
python3 scripts/benchmark_dla.py --model=resnet18_int8_dla --batch_size=$BATCH_SIZE --use_cache --dla_core=1 --distributed --rank=1 --world_size=2