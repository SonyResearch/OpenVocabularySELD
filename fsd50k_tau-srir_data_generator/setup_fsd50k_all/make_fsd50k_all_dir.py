import os
import shutil
import pandas as pd
import glob
import tqdm


path_glob = "../data_fsd50k_tau-srir/fsd50k/FSD50K.*_audio/*.wav"
list_glob = glob.glob(path_glob)

pd.set_option("display.max_colwidth", 100)
path_ground_truth_dev = "../data_fsd50k_tau-srir/fsd50k/FSD50K.ground_truth/dev.csv"
df_gt_dev = pd.read_csv(path_ground_truth_dev)
path_ground_truth_eval = "../data_fsd50k_tau-srir/fsd50k/FSD50K.ground_truth/eval.csv"
df_gt_eval = pd.read_csv(path_ground_truth_eval)

dir_all = "../data_fsd50k_tau-srir/fsd50k/fsd50k_all"

for line in tqdm.tqdm(list_glob):
    fname_line = os.path.splitext(os.path.basename(line))[0]

    if "dev_audio" in line:
        init_label = df_gt_dev[df_gt_dev["fname"] == int(fname_line)]["labels"].to_string(index=False).split(",")[0]  # initial one of ground truth labels
        path_dst = os.path.join(dir_all, init_label.lower(), "train", init_label, fname_line + ".wav")
    elif "eval_audio" in line:
        init_label = df_gt_eval[df_gt_eval["fname"] == int(fname_line)]["labels"].to_string(index=False).split(",")[0]
        path_dst = os.path.join(dir_all, init_label.lower(), "test", init_label, fname_line + ".wav")

    dir_dst = os.path.dirname(path_dst)
    os.makedirs(dir_dst, exist_ok=True)

    shutil.copyfile(line, path_dst)
