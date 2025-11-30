#!/bin/bash

cd DeT
python pytracking/run_tracker.py dimp DeT_DiMP50_Max_original --dataset_name depthtrack_train --input_dtype rgbcolormap \
        --save_features "/mnt/data/depthtrack/train"