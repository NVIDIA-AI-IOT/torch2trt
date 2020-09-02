#!/bin/bash

TAG=$1

python3 scripts/dump_converters.py > docs/converters.md

mike deploy $TAG --push
