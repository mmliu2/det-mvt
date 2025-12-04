# #!/bin/bash

cd DeT/ltr
python run_training.py mvt DeT_MVT_Max_KD001
# python run_training.py mvt DeT_MVT_Max_KD002
# python run_training.py mvt DeT_MVT_Max_CRD001
# python run_training.py mvt DeT_MVT_Max_CRD002
# python run_training.py mvt DeT_MVT_Max_KD001_CRD001
python run_training.py mvt DeT_MVT_Max_cf_KD001
# python run_training.py mvt DeT_MVT_Max_cf_KD002
# python run_training.py mvt DeT_MVT_Max_cf_CRD001
# python run_training.py mvt DeT_MVT_Max_cf_CRD002
# python run_training.py mvt DeT_MVT_Max_KD001_cf_CRD001
### python run_training.py mvt DeT_MVT_Max_KD002_CRD002 # skip
# baselines
# python run_training.py mvt DeT_MVT_Max
cd ../..

# evaluate on DepthTrack
cd DeT
# python pytracking/run_tracker.py mvt DeT_MVT_Max_KD001 --dataset_name depthtrack --input_dtype rgbcolormap --threads 4
# python pytracking/run_tracker.py mvt DeT_MVT_Max_KD002 --dataset_name depthtrack --input_dtype rgbcolormap --threads 4
# python pytracking/run_tracker.py mvt DeT_MVT_Max_CRD001 --dataset_name depthtrack --input_dtype rgbcolormap --threads 4
# python pytracking/run_tracker.py mvt DeT_MVT_Max_CRD002 --dataset_name depthtrack --input_dtype rgbcolormap --threads 4
# python pytracking/run_tracker.py mvt DeT_MVT_Max_KD001_CRD001 --dataset_name depthtrack --input_dtype rgbcolormap --threads 4
### python pytracking/run_tracker.py mvt DeT_MVT_Max_KD002_CRD002 --dataset_name depthtrack --input_dtype rgbcolormap --threads 4
# baselines
# python pytracking/run_tracker.py dimp DeT_DiMP50_Max --dataset_name depthtrack --input_dtype rgbcolormap --threads 4
# python pytracking/run_tracker.py mvt DeT_MVT_Max --dataset_name depthtrack --input_dtype rgbcolormap --threads 4
cd ..

# evaluate on VOT
cd DeT
# python pytracking/run_tracker.py mvt DeT_MVT_Max_KD001 --dataset_name votrgbd22 --input_dtype rgbcolormap --threads 4
# python pytracking/run_tracker.py mvt DeT_MVT_Max_KD002 --dataset_name votrgbd22 --input_dtype rgbcolormap --threads 4
# python pytracking/run_tracker.py mvt DeT_MVT_Max_CRD001 --dataset_name votrgbd22 --input_dtype rgbcolormap --threads 4
# python pytracking/run_tracker.py mvt DeT_MVT_Max_CRD002 --dataset_name votrgbd22 --input_dtype rgbcolormap --threads 4
# python pytracking/run_tracker.py mvt DeT_MVT_Max_KD001_CRD001 --dataset_name votrgbd22 --input_dtype rgbcolormap --threads 4
### python pytracking/run_tracker.py mvt DeT_MVT_Max_KD002_CRD002 --dataset_name votrgbd22 --input_dtype rgbcolormap --threads 4
# baselines
# python pytracking/run_tracker.py dimp DeT_DiMP50_Max_original --dataset_name votrgbd22 --input_dtype rgbcolormap --threads 4
# python pytracking/run_tracker.py dimp DeT_DiMP50_Max --dataset_name votrgbd22 --input_dtype rgbcolormap --threads 4
# python pytracking/run_tracker.py mvt DeT_MVT_Max --dataset_name votrgbd22 --input_dtype rgbcolormap --threads 4
cd ..

# analyze DepthTrack results
cd DeT
# python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/mvt/DeT_MVT_Max_KD001 --sequences_dir ../../data/depthtrack/test/
# python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/mvt/DeT_MVT_Max_KD002 --sequences_dir ../../data/depthtrack/test/
# python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/mvt/DeT_MVT_Max_CRD001 --sequences_dir ../../data/depthtrack/test/
# python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/mvt/DeT_MVT_Max_CRD002 --sequences_dir ../../data/depthtrack/test/  
# python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/mvt/DeT_MVT_Max_KD001_CRD001 --sequences_dir ../../data/depthtrack/test/
### python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/mvt/DeT_MVT_Max_KD002_CRD002 --sequences_dir ../../data/depthtrack/test/
# baselines
# python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/dimp/DeT_DiMP50_Max_original_test --sequences_dir ../../data/depthtrack/test/
# python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/dimp/DeT_DiMP50_Max --sequences_dir ../../data/depthtrack/test/
# python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/mvt/DeT_MVT_Max --sequences_dir ../../data/depthtrack/test/
cd ..

# analyze VOT results
cd DeT
# python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/mvt/DeT_MVT_Max_KD001 --sequences_dir ../../data/vot/sequences/
# python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/mvt/DeT_MVT_Max_KD002 --sequences_dir ../../data/vot/sequences/
# python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/mvt/DeT_MVT_Max_CRD001 --sequences_dir ../../data/vot/sequences/
# python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/mvt/DeT_MVT_Max_CRD002 --sequences_dir ../../data/vot/sequences/
# python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/mvt/DeT_MVT_Max_KD001_CRD001 --sequences_dir ../../data/vot/sequences/
### python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/mvt/DeT_MVT_Max_KD002_CRD002 --sequences_dir ../../data/vot/sequences/
# baselines
# python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/dimp/DeT_DiMP50_Max_original --sequences_dir ../../data/vot/sequences/
# python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/dimp/DeT_DiMP50_Max --sequences_dir ../../data/vot/sequences/
# python pytracking/depthtrack_results.py --pred_dir pytracking/tracking_results/mvt/DeT_MVT_Max --sequences_dir ../../data/vot/sequences/
cd ..
