#!/usr/bin/env sh
mode=$1
datatype=$2
runname=$3

cmd="python3 -m project.compare -m \"$mode\" -D \"$datatype\""

if [ -n "$runname" ]; then
    cmd="$cmd -R $runname"
fi

echo Running "$cmd"

eval "$cmd"

