#!/bin/bash

<<<<<<< HEAD
GITHUB=$1
TAG=$2

python3 scripts/dump_converters.py --github=$GITHUB --tag=$TAG > docs/converters.md
=======
TAG=$1

# python3 scripts/dump_converters.py --tag=$TAG > docs/converters.md

mike deploy $TAG
>>>>>>> 4933faf05cd9e5d9aef2b23e7fe692bfda5fbc06
