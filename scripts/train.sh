#!/usr/bin/env sh

set -e

RUN=$1
CONFIG=$2
RESUME=$3

if [ -z "$RUN" ] || [ -z "$CONFIG" ]; then
    echo "Usage: ./train.sh <model> <config.json> [resume_file]"
    exit 1
fi

CMD="python -m project.deep_learning.train_neural_network -R \"$RUN\" -C \"$CONFIG\""

if [ -n "$RESUME" ]; then
    CMD="$CMD -r $RESUME"
fi

echo "Running: $CMD"
eval "$CMD"
