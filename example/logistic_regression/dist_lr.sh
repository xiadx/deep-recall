#!/usr/bin/env bash

python="/Users/xdx/anaconda3/envs/tf1.x/bin/python2.7"

job_path=$(dirname $(dirname "$PWD"))

${python} dist_lr.py --ps_hosts=127.0.0.1:9100 --worker_hosts=127.0.0.1:9200 --task_index=0 --job_name=ps --files=${job_path}/data/sample.csv --checkpointDir=${job_path}/data/models
${python} dist_lr.py --ps_hosts=127.0.0.1:9100 --worker_hosts=127.0.0.1:9200 --task_index=0 --job_name=worker --files=${job_path}/data/sample.csv --checkpointDir=${job_path}/data/models