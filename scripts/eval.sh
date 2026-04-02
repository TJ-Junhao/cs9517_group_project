#!/usr/bin/env sh

RUN=$1
CONFIG=$2

cmd="python -m project.evaluate -R \"$RUN\" -C \"$CONFIG\" -m test"


echo Running "$cmd"
eval "$cmd"

cmd="python -m project.evaluate -R \"$RUN\" -C \"$CONFIG\" -m train"

echo Running "$cmd"
eval "$cmd"

cmd="python -m project.evaluate -R \"$RUN\" -C \"$CONFIG\" -m validation"

echo Running "$cmd"
eval "$cmd"
