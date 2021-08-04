#!/usr/bin/env bash

python="/Users/xdx/anaconda3/envs/tf1.x/bin/python2.7"

job_path=$(dirname $(dirname "$PWD"))

${python} hello_world.py --files ${job_path}/data/sample.csv