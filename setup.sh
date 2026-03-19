#!/bin/sh

if command -v uv 2> /dev/null
then
    echo Running uv sync...
    uv sync
else
    echo "Please install uv to setup the environment"
    echo "Hint: Run 'pip install uv'"
fi
