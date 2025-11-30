#!/bin/bash

cd DeT
# python pytracking/run_tracker.py dimp DeT_DiMP50_Max --dataset_name depthtrack --input_dtype rgbcolormap
python pytracking/run_tracker.py dimp DeT_DiMP50_Max --dataset_name depthtrack --input_dtype rgbcolormap --threads 4

# delete this
python pytracking/depthtrack_results.py \
        --sequences_dir ../../data/depthtrack/test/ \
        --pred_dir pytracking/tracking_results/dimp/DeT_DiMP50_Max