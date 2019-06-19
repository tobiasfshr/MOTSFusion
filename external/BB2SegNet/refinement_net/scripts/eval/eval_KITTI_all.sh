#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Usage ${0} config"
    exit
fi

config=$1

grep forwarded $config | my_cut.py 2 | sed s/,//g | print_mean.py
