#!/bin/bash

GITHUB=$1
TAG=$2

python3 scripts/dump_converters.py --github=$GITHUB --tag=$TAG > docs/converters.md
