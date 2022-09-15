#!/bin/bash

git clone https://github.com/NVIDIA/TensorRT.git /tmp/TensorRT/

parentdir="$(dirname "$(pwd)")"
patch="examples/contrib/quantization_aware_training/utils/torch2trt_qat.patch"
patch_file="$parentdir/$patch"

pushd /tmp/TensorRT
    cp $patch_file .
    
    ## Official tags from NVIDIA/TensorRT can't be used as `tools/` folder is not part of official tags

    git checkout e5f9ead4a4826cc774325720a26dbf4ec47203ea  ## NVIDIA/TensorRT master on Sept 14, 2022
    git sparse-checkout init --cone 
    git sparse-checkout set /tools/pytorch-quantization/
    git apply --whitespace=fix torch2trt_qat.patch
    cd tools/pytorch-quantization/
    python setup.py install
popd

pushd $parentdir
    python3 setup.py install --contrib
popd


