#!/bin/bash

TAG=$1

mike set-default $TAG
mike serve --dev-addr=0.0.0.0:8000