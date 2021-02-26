#!/bin/bash

BATCH_SIZE=1

# resnet18

# fp16
python3 scripts/benchmark_dla.py --model=resnet18_fp16_gpu --batch_size=$BATCH_SIZE --use_cache
python3 scripts/benchmark_dla.py --model=resnet18_fp16_dla --batch_size=$BATCH_SIZE --use_cache

# int8
python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu --batch_size=$BATCH_SIZE --use_cache
python3 scripts/benchmark_dla.py --model=resnet18_int8_dla --batch_size=$BATCH_SIZE --use_cache

# hybrid fp16
python3 scripts/benchmark_dla.py --model=resnet18_fp16_gpu_dla1 --batch_size=$BATCH_SIZE --use_cache
python3 scripts/benchmark_dla.py --model=resnet18_fp16_gpu_dla12 --batch_size=$BATCH_SIZE --use_cache
python3 scripts/benchmark_dla.py --model=resnet18_fp16_gpu_dla123 --batch_size=$BATCH_SIZE --use_cache

# hybrid int8
python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu_dla1 --batch_size=$BATCH_SIZE --use_cache
python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu_dla12 --batch_size=$BATCH_SIZE --use_cache
python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu_dla123 --batch_size=$BATCH_SIZE --use_cache

# resnet50 

# fp16
python3 scripts/benchmark_dla.py --model=resnet50_fp16_gpu --batch_size=$BATCH_SIZE --use_cache
python3 scripts/benchmark_dla.py --model=resnet50_fp16_dla --batch_size=$BATCH_SIZE --use_cache

# int8
python3 scripts/benchmark_dla.py --model=resnet50_int8_gpu --batch_size=$BATCH_SIZE --use_cache
python3 scripts/benchmark_dla.py --model=resnet50_int8_dla --batch_size=$BATCH_SIZE --use_cache

# hybrid fp16
python3 scripts/benchmark_dla.py --model=resnet50_fp16_gpu_dla1 --batch_size=$BATCH_SIZE --use_cache
python3 scripts/benchmark_dla.py --model=resnet50_fp16_gpu_dla12 --batch_size=$BATCH_SIZE --use_cache
python3 scripts/benchmark_dla.py --model=resnet50_fp16_gpu_dla123 --batch_size=$BATCH_SIZE --use_cache

# hybrid int8
python3 scripts/benchmark_dla.py --model=resnet50_int8_gpu_dla1 --batch_size=$BATCH_SIZE --use_cache
python3 scripts/benchmark_dla.py --model=resnet50_int8_gpu_dla12 --batch_size=$BATCH_SIZE --use_cache
python3 scripts/benchmark_dla.py --model=resnet50_int8_gpu_dla123 --batch_size=$BATCH_SIZE --use_cache


# BATCH_SIZE=2

# resnet18

# # fp16
# python3 scripts/benchmark_dla.py --model=resnet18_fp16_gpu --batch_size=$BATCH_SIZE
# python3 scripts/benchmark_dla.py --model=resnet18_fp16_dla --batch_size=$BATCH_SIZE

# # int8
# python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu --batch_size=$BATCH_SIZE
# python3 scripts/benchmark_dla.py --model=resnet18_int8_dla --batch_size=$BATCH_SIZE

# # hybrid fp16
# python3 scripts/benchmark_dla.py --model=resnet18_fp16_gpu_dla1 --batch_size=$BATCH_SIZE
# python3 scripts/benchmark_dla.py --model=resnet18_fp16_gpu_dla12 --batch_size=$BATCH_SIZE
# python3 scripts/benchmark_dla.py --model=resnet18_fp16_gpu_dla123 --batch_size=$BATCH_SIZE

# # hybrid int8
# python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu_dla1 --batch_size=$BATCH_SIZE
# python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu_dla12 --batch_size=$BATCH_SIZE
# python3 scripts/benchmark_dla.py --model=resnet18_int8_gpu_dla123 --batch_size=$BATCH_SIZE

# # resnet50 

# # fp16
# python3 scripts/benchmark_dla.py --model=resnet50_fp16_gpu --batch_size=$BATCH_SIZE
# python3 scripts/benchmark_dla.py --model=resnet50_fp16_dla --batch_size=$BATCH_SIZE

# # int8
# python3 scripts/benchmark_dla.py --model=resnet50_int8_gpu --batch_size=$BATCH_SIZE
# python3 scripts/benchmark_dla.py --model=resnet50_int8_dla --batch_size=$BATCH_SIZE

# # hybrid fp16
# python3 scripts/benchmark_dla.py --model=resnet50_fp16_gpu_dla1 --batch_size=$BATCH_SIZE
# python3 scripts/benchmark_dla.py --model=resnet50_fp16_gpu_dla12 --batch_size=$BATCH_SIZE
# python3 scripts/benchmark_dla.py --model=resnet50_fp16_gpu_dla123 --batch_size=$BATCH_SIZE

# # hybrid int8
# python3 scripts/benchmark_dla.py --model=resnet50_int8_gpu_dla1 --batch_size=$BATCH_SIZE
# python3 scripts/benchmark_dla.py --model=resnet50_int8_gpu_dla12 --batch_size=$BATCH_SIZE
# python3 scripts/benchmark_dla.py --model=resnet50_int8_gpu_dla123 --batch_size=$BATCH_SIZE