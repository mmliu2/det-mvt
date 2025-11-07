#!/bin/bash

# create local file before running experiments

cd MVT
python3 tracking/create_default_local_file.py \
    --workspace_dir . \
    --data_dir ../data \
    --save_dir ../output
