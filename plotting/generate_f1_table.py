import os
import ast


MODEL_NAME_TO_DISPLAY_NAME = [
    ('DeT_DiMP50_Max_original', 'DeT-DiMP50-original'),
    ('DeT_DiMP50_Max', 'DeT-DiMP50'),
    ('DeT_MVT_Max', 'DeT-MVT'),
    ('DeT_MVT_Max_KD001', 'DeT-MVT+rsKD'),
    ('DeT_MVT_Max_KD002', 'DeT-MVT+2rsKD'),
    ('DeT_MVT_Max_CRD001', 'DeT-MVT+rsCRD'),
    ('DeT_MVT_Max_CRD001', 'DeT-MVT+2rsCRD'),
    ('DeT_MVT_Max_KD001_CRD001', 'DeT-MVT+rsKD+rsCRD'),
    ('DeT_MVT_Max_cf_KD001', 'DeT-MVT+cfKD'),
    ('DeT_MVT_Max_cf_KD002', 'DeT-MVT+cf2KD'),
    ('DeT_MVT_Max_cf_CRD001', 'DeT-MVT+cfCRD'),
    ('DeT_MVT_Max_cf_CRD001', 'DeT-MVT+2cfCRD'),
    ('DeT_MVT_Max_cf_KD001_CRD001', 'DeT-MVT+cfKD+cfCRD'),
]

def generate_latex_table(metrics_dict, caption="Model Performance", label="tab:results"):
    """
    metrics_dict format:
    {model_name:
        {
            "params": float,
            "depthtrack": {
                conf_thresh: {
                    "seq_avg": {"f1":..., "precision":..., "recall":...},
                    "frame_avg": {...}
                },
                ...
            },
            "vot": {
                conf_thresh: {...},
                ...
            }
        }
    }
    """

    # -------- helper: pick best confidence threshold per dataset -------- #
    def select_best_threshold(dataset_dict):
        best_t = None
        best_f1 = -1
        best_entry = None

        for t, stats in dataset_dict.items():
            f1 = stats["seq_avg"]["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                best_entry = stats["seq_avg"]

        return best_t, best_entry

#     # -------- LaTeX header -------- #
#     latex = r"""\begin{table}[H]
# \centering
# \begin{tabular}{l|ccc|ccc|c}
# \hline
# \textbf{Model} & \multicolumn{3}{c|}{\textbf{DepthTrack}} & \multicolumn{3}{c|}{\textbf{VOT-RGBD22}} & \textbf{Params (M)} \\
#  & F1 & Pr & Re & F1 & Pr & Re & \\
# \hline
# """
    latex = r""

    # -------- each model: extract best DT + VOT results -------- #
    # for model, datasets in metrics_dict.items():
    for model_name, display_name in MODEL_NAME_TO_DISPLAY_NAME:

        # check if model exists
        datasets = metrics_dict.get(model_name)
        if datasets is None:
            # model missing → all metrics are "--"
            dt_f1 = dt_pr = dt_re = None
            vt_f1 = vt_pr = vt_re = None
        else:
            print('found', display_name)
            # params

            # depthtrack block
            if "depthtrack" in datasets and datasets["depthtrack"]:
                _, dt = select_best_threshold(datasets["depthtrack"])
                dt_f1 = dt["f1"]
                dt_pr = dt["precision"]
                dt_re = dt["recall"]
            else:
                dt_f1 = dt_pr = dt_re = None
                if model_name == 'DeT_DiMP50_Max_original':
                    dt_f1 = 0.660
                    dt_pr = 0.777
                    dt_re = 0.579

            # vot block
            if "vot" in datasets and datasets["vot"]:
                _, vt = select_best_threshold(datasets["vot"])
                vt_f1 = vt["f1"]
                vt_pr = vt["precision"]
                vt_re = vt["recall"]
            else:
                vt_f1 = vt_pr = vt_re = None

        # helper for formatting
        def fmt(x):
            return f"{x:.3f}" if isinstance(x, (float, int)) else "--"

        latex += (
            f"{display_name} & "
            f"{fmt(dt_f1)} & {fmt(dt_pr)} & {fmt(dt_re)} & "
            f"{fmt(vt_f1)} & {fmt(vt_pr)} & {fmt(vt_re)} & "
            f"12.9 \\\\\n"
        )

#     # -------- footer -------- #
#     latex += r"""\hline
# \end{tabular}
# \caption{""" + caption + r"""}
# \label{""" + label + r"""}
# \end{table}
# """

    return latex


def list_grandchildren_txt(d):
    if not os.path.exists(d): return []

    grandchildren = []
    for child in os.listdir(d):
        child_path = os.path.join(d, child)
        if os.path.isdir(child_path):
            for grandchild in os.listdir(child_path):
                grandchild_path = os.path.join(child_path, grandchild)
                if os.path.isfile(grandchild_path) and grandchild_path.endswith(".txt"):
                    grandchildren.append(grandchild_path)
    return grandchildren

def parse_results(results_txt):
    results_all_thresholds = {}
    with open(results_txt, 'r') as f:
        for line in f.readlines()[::-1]:
            if ('{' not in line or 'F1=' not in line) and results_all_thresholds:
                break
            dict_start_idx = line.find('{')
            dict_str = line[dict_start_idx:]

            confidence_threshold = float(line.split(';')[0][-3:])
            # print(confidence_threshold)
            results = ast.literal_eval(dict_str)
            results_all_thresholds[confidence_threshold] = results

    return results_all_thresholds

def get_metrics():
    metrics = {}

    # {model name:
    #    {depthtrack or vot:
    #       {confidence threshold:
    #           {seq_avg or frame_avg:
    #               {f1:
    #                pr:
    #                re:

    depthtrack_rs_dir = '/mnt/det-mvt/DeT/pytracking/tracking_results_rs_depthtrack'
    vot_rs_dir = '/mnt/det-mvt/DeT/pytracking/tracking_results_rs_vot'
    depthtrack_cf_dir = '/mnt/det-mvt/DeT/pytracking/tracking_results'
    vot_cf_dir = '/mnt/det-mvt/DeT/pytracking/tracking_results_cf_vot'

    depthtrack_results_txts = list_grandchildren_txt(depthtrack_rs_dir) + list_grandchildren_txt(depthtrack_cf_dir)
    vot_results_txts = list_grandchildren_txt(vot_rs_dir) + list_grandchildren_txt(vot_cf_dir)
    
    for dataset_name, results_txts in zip(['depthtrack', 'vot'], [depthtrack_results_txts, vot_results_txts]):
        for results_txt in results_txts:
            model_name = os.path.basename(results_txt)[:-4]

            if model_name not in metrics:
                metrics[model_name] = {}
            
            print('parsing', results_txt)
            model_results = parse_results(results_txt)

            metrics[model_name][f'{dataset_name}'] = model_results
    return metrics


if __name__ == '__main__':

    metrics = get_metrics()
    print()

    latex_code = generate_latex_table(metrics, caption="Classification Results", label="tab:cls-results")
    print()
    print(latex_code)