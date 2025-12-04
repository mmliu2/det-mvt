import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import cv2
import subprocess
from tqdm import tqdm
import math
from glob import glob
import numpy as np
import csv
from datetime import datetime


USE_VISIBLE_TAGS = False



def box_to_mask(box, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)

    if math.isnan(box[0]): 
        return mask

    x, y, w, h = box
    x1 = max(int(x), 0)
    y1 = max(int(y), 0)
    x2 = min(int(x + w), W)
    y2 = min(int(y + h), H)
    mask[y1:y2, x1:x2] = 1

    return mask

def precision_recall_f1(pred_box, gt_box, H, W, confidence, confidence_threshold, is_visible_tag):
    # object not visible in ground truth
    if is_visible_tag is not None:
        not_visible = math.isnan(gt_box[0]) or not is_visible_tag 
    else:
        not_visible = math.isnan(gt_box[0])

    if not_visible: 
        if confidence > confidence_threshold:
            return 0.0, 0.0, 0.0
        else:   
            return 1.0, 1.0, 1.0

    pred_mask = box_to_mask(pred_box, H, W)
    gt_mask = box_to_mask(gt_box, H, W)

    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()

    if confidence > confidence_threshold:
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0 
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0 # harmonic mean
    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    return precision, recall, f1

def compute_avg_metrics(pr_list, re_list, f1_list):
    pr_array = [np.array(seq) for seq in pr_list]
    re_array = [np.array(seq) for seq in re_list]
    f1_array = [np.array(seq) for seq in f1_list]
    
    pr_seq_avg = np.mean([seq.mean() for seq in pr_array])
    re_seq_avg = np.mean([seq.mean() for seq in re_array])
    # f1_seq_avg = np.mean([seq.mean() for seq in f1_array])
    f1_seq_avg = 2 * pr_seq_avg * re_seq_avg / (pr_seq_avg + re_seq_avg) if (pr_seq_avg + re_seq_avg) > 0 else 0.0
    seq_avg = {
        'precision': pr_seq_avg,
        'recall':    re_seq_avg,
        'f1':        f1_seq_avg,
    }

    all_pr = np.concatenate(pr_array)
    all_re = np.concatenate(re_array)
    all_f1 = np.concatenate(f1_array)

    pr_all_avg = all_pr.mean()
    re_all_avg = all_re.mean()
    # f1_all_avg = all_f1.mean()
    f1_all_avg = 2 * pr_all_avg * re_all_avg / (pr_all_avg + re_all_avg) if (pr_all_avg + re_all_avg) > 0 else 0.0

    frame_avg = {
        'precision': pr_all_avg,
        'recall':    re_all_avg, 
        'f1':        f1_all_avg,
    }

    return {'seq_avg': seq_avg, 'frame_avg': frame_avg}

def sequence_results_to_metrics(sequences_dir, pred_dir):
    metrics_path = os.path.join(pred_dir, f'../{os.path.basename(pred_dir)}.txt')
    with open(metrics_path, 'a') as f:
        f.write('\n')
        f.write(str(datetime.today()))
    

    # for confidence_threshold in [0.4, 0.6, 0.2, 0.8, 0.0, 1.0]:
    for confidence_threshold in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
    # for confidence_threshold in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        print('Evaluating with confidence threshold:', confidence_threshold)
        
        pr_list = []
        re_list = []
        f1_list = []

        sequence_names = sorted(os.listdir(sequences_dir))
        # for sequence_name in sequence_names:
        for sequence_name in tqdm(sequence_names, total=len(sequence_names)):
            gt_dir = os.path.join(sequences_dir, sequence_name)
            im_dir = os.path.join(gt_dir, 'color')

            gt_file = os.path.join(gt_dir, 'groundtruth.txt')
            pred_file = os.path.join(pred_dir, f'{sequence_name}/{sequence_name}_001.txt')
            pred_conf_file = os.path.join(pred_dir, f'{sequence_name}/{sequence_name}_001_confidence.value')

            # import pdb; pdb.set_trace()

            if not os.path.isfile(pred_file): 
                # print('  skipping', sequence_name)
                continue
            # print(sequence_name)
            
            # get image files
            im_files = sorted([os.path.join(im_dir, f) for f in os.listdir(im_dir)
                            if f.lower().endswith(('.jpg', '.png'))])
            
            num_frames = len(im_files)


            if USE_VISIBLE_TAGS: # TODO
                occlusion_file = os.path.join(gt_dir, 'full-occlusion.tag')
                out_of_view_file = os.path.join(gt_dir, 'out-of-frame.tag')

                with open(occlusion_file, 'r') as f:
                    occ = [int(v) for v in f.readlines()]
                with open(out_of_view_file, 'r') as f:
                    oov = [int(v) for v in f.readlines()]
                target_visible = [(o == 0 and v == 0) for o, v in zip(occ, oov)]

                with open(out_of_view_file, 'r') as f:
                    oov = [int(v) for v in f.readlines()]
                target_visible = [v == 0 for v in oov]
            else:
                target_visible = [None for v in range(num_frames)]

            # load boxes
            pred_boxes = []
            with open(pred_file, 'r') as f:
                for line in f:
                    x, y, w, h = map(float, line.strip().split(','))
                    pred_boxes.append([x, y, w, h])
            pred_conf = []
            with open(pred_conf_file, 'r') as f:
                for line in f:
                    conf = float(line.strip())
                    pred_conf.append(conf)

            gt_boxes = []
            with open(gt_file, 'r') as f:
                for line in f:
                    x, y, w, h = map(float, line.strip().split(','))
                    gt_boxes.append([x, y, w, h])

            seq_pr_list = []
            seq_re_list = []
            seq_f1_list = []

            im = cv2.imread(im_files[0])
            H, W, _ = im.shape
            # for i in tqdm(range(0, num_frames), total=num_frames): 
            for i in range(0, num_frames): 
                precision, recall, f1 = precision_recall_f1(pred_boxes[i], gt_boxes[i], H, W, 
                                                            pred_conf[i], confidence_threshold, target_visible[i])

                seq_pr_list.append(precision)
                seq_re_list.append(recall)
                seq_f1_list.append(f1)
                
            # print('average f1 for sequence:', np.mean(np.array(seq_f1_list)))

            pr_list.append(seq_pr_list)
            re_list.append(seq_re_list)
            f1_list.append(seq_f1_list)

        metrics = compute_avg_metrics(pr_list, re_list, f1_list)
        metrics_summary = f"F1={metrics['seq_avg']['f1']:.3f}, Pr={metrics['seq_avg']['precision']:.3f}, Re={metrics['seq_avg']['recall']:.3f}"
        print(metrics_summary)
        
        with open(metrics_path, 'a') as f:
            f.write(str(confidence_threshold) + "; " + metrics_summary + "; " + str(metrics) + '\n')

    
def main():
    # render video with bounding boxes on depthtrack color images
    parser = argparse.ArgumentParser(description='Convert single example sequence to video.')
    # parser.add_argument('--ffmpeg_path', type=str, default='/usr/local/bin/ffmpeg/ffmpeg-git-20240629-amd64-static/ffmpeg', help='Path to FFmpeg.')
    parser.add_argument('--sequences_dir', type=str, default='../../data/depthtrack/test/', help='Directory containing sequences.')
    parser.add_argument('--pred_dir', type=str, default='pytracking/tracking_results/dimp/DeT_DiMP50_Max_original_test', help='Directory containing predictions text file.')

    args = parser.parse_args()

    sequence_results_to_metrics(args.sequences_dir, args.pred_dir)


if __name__ == '__main__':
    main()
