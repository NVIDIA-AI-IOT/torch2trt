#!/bin/bash -exu

PATCH_DIR="examples/contrib/pre_py3.7/"
PATCH_FILES=(
    "fix-getitem.patch"
)

for patch_file in "${PATCH_FILES[@]}"; do
    patch_file="${PATCH_DIR}""${patch_file}"
    git apply "${patch_file}"
done

python3 setup.py install
