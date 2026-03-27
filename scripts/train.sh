#!/usr/bin/env sh

set -e

MODEL=$1
CONFIG=$2
RESUME=$3

if [ -z "$MODEL" ] || [ -z "$CONFIG" ]; then
    echo "Usage: ./train.sh <model> <config.json> [resume_file]"
    exit 1
fi

CMD="python -m project.train \"$MODEL\" -C \"$CONFIG\""

if [ -n "$RESUME" ]; then
    CMD="$CMD -r $RESUME"
fi

echo "Running: $CMD"
eval "$CMD"
