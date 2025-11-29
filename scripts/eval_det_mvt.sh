#!/bin/bash

cd DeT
python pytracking/run_tracker.py mvt DeT_MVT_Max --dataset_name depthtrack --input_dtype rgbcolormap
# python pytracking/run_tracker.py mvt DeT_MVT_Max --dataset_name depthtrack --input_dtype rgbcolormap --threads 4