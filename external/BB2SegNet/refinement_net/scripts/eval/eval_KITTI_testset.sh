#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Usage ${0} config"
    exit
fi

config=$1

count=$(grep forwarded $config | wc -l)
if [ $count -ne 934 ]; then
    echo "incorrect number of ious"
    exit
fi

grep forwarded $config | my_cut.py 2 | sed s/,//g | head -651 | print_mean.py
