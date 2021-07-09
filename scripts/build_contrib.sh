#!/bin/bash

git clone https://github.com/NVIDIA/TensorRT.git /tmp/TensorRT/

parentdir="$(dirname "$(pwd)")"
patch="examples/contrib/quantization_aware_training/utils/pytorch_nvidia_quantization.patch"
patch_file="$parentdir/$patch"

pushd /tmp/TensorRT
    cp $patch_file .
    git checkout e724d31ab84626ca334b4284703b5048eb698c98  ## keeping this for versioning control
    git sparse-checkout init --cone 
    git sparse-checkout set /tools/pytorch-quantization/
    git apply --reject --whitespace=fix pytorch_nvidia_quantization.patch
    cd tools/pytorch-quantization/
    python setup.py install
popd

pushd $parentdir
    python3 setup.py install --plugins --contrib
popd


