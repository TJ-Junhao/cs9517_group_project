#!/usr/bin/env sh

MODEL=$1
CONFIG=$2
MODE=$3

python -m project.evaluate \
    "$MODEL" \
    -C "$CONFIG" \
    -m "$MODE"

