#!/bin/bash

# generates model cache for use with multiprocessing benchmark
BATCH_SIZE=1

# NOTE: currently we can't use DLA1 with deserialized engines.  The reason is likely because we need to set the DLA in runtime,
# but TensorRT python API does not expose this feature from the TensorRT C++ API.

# INT8

# 2 - 0
python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu --batch_size=$BATCH_SIZE --use_cache --distributed --rank=0 --world_size=2 &
python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu --batch_size=$BATCH_SIZE --use_cache --distributed --rank=1 --world_size=2

# 1 - 1
python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu --batch_size=$BATCH_SIZE --use_cache --distributed --rank=0 --world_size=2 &
python3 scripts/benchmark_dla.py --model=resnet18_int8_dla --batch_size=$BATCH_SIZE --use_cache --dla_core=0 --distributed --rank=1 --world_size=2 

# 0 - 2 (skip use-cache because of missing setDLACore when deserializing engine)
python3 scripts/benchmark_dla.py --model=resnet18_int8_dla --batch_size=$BATCH_SIZE --dla_core=0 --distributed --rank=0 --world_size=2 &
python3 scripts/benchmark_dla.py --model=resnet18_int8_dla --batch_size=$BATCH_SIZE --dla_core=1 --distributed --rank=1 --world_size=2

# FP16

# 2 - 0
python3 scripts/benchmark_dla.py --model=resnet18_fp16_gpu --batch_size=$BATCH_SIZE --use_cache --distributed --rank=0 --world_size=2 &
python3 scripts/benchmark_dla.py --model=resnet18_fp16_gpu --batch_size=$BATCH_SIZE --use_cache --distributed --rank=1 --world_size=2

# 1 - 1
python3 scripts/benchmark_dla.py --model=resnet18_fp16_gpu --batch_size=$BATCH_SIZE --use_cache --distributed --rank=0 --world_size=2 &
python3 scripts/benchmark_dla.py --model=resnet18_fp16_dla --batch_size=$BATCH_SIZE --use_cache --dla_core=0 --distributed --rank=1 --world_size=2 

# 0 - 2
python3 scripts/benchmark_dla.py --model=resnet18_fp16_dla --batch_size=$BATCH_SIZE --dla_core=0 --distributed --rank=0 --world_size=2 &
python3 scripts/benchmark_dla.py --model=resnet18_fp16_dla --batch_size=$BATCH_SIZE --dla_core=1 --distributed --rank=1 --world_size=2