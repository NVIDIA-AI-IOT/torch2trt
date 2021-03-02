#!/bin/bash

TAG=$1

python3 scripts/dump_converters.py --tag=$TAG > docs/converters.md

mike deploy $TAG --push
