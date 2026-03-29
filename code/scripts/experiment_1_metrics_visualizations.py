# Experiment 1: Visualize the metrics for different methods and datasets through the change of label weights
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data import data_prep
import os
import argparse
import pacmap
from evaluation import soft_jaccard_similarity,faster_centroid_triplet_eval,random_triplet_eval,faster_knn_eval_series,record_loss_pacmap, record_loss_umap, record_NCVis_loss, record_tSNE_loss, record_infonce_loss
import tqdm

# Results directory: set RESULTS_DIR environment variable or defaults to ./results
RESULTS_DIR = os.environ.get('RESULTS_DIR', './results')

parser = argparse.ArgumentParser(description='Visualize the results of experiment 2')
parser.add_argument('--dataset', type=str, default='MNIST', help='dataset name')
parser.add_argument('--n_repeats', type=int, default=5, help='number of repeats')
parser.add_argument('--method', type=str, default="pacmap", help='method name')
parser.add_argument('--task_type', type=str, default="axis", help='task type')
parser.add_argument('--metric_type', type=str, default="Jaccard", help='metric type')

args = parser.parse_args()

x_axis= [0.0, 0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
                0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
                0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
                1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,
                20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,
                100.0]

    
if args.method != "pacmap":
    x_axis = [round(i*10000, 1) for i in x_axis]


if (args.metric_type != "Jaccard") and (args.metric_type != "Loss"):
    # load the score
    score_lst = np.load(f"{RESULTS_DIR}/scores/{args.dataset}/{args.method}/{args.task_type}/score_list_{args.metric_type}.npy", allow_pickle=True)
    if len(score_lst.shape) == 1:
        score_lst = score_lst.reshape(-1, args.n_repeats, 1)
    else:
        score_lst = score_lst.reshape(-1, args.n_repeats, score_lst.shape[1])

elif ((args.metric_type == "Loss") and (args.method == "pacmap")):
        score_lst = []
        for i in x_axis:
            temp_lst = []
            for j in range(args.n_repeats):
                # load the loss
                NN_loss = np.load(f"{RESULTS_DIR}/losses/{args.dataset}/{args.method}/{args.task_type}/NN_loss_{i}_{j}.npy", allow_pickle=True)
                MN_loss = np.load(f"{RESULTS_DIR}/losses/{args.dataset}/{args.method}/{args.task_type}/MN_loss_{i}_{j}.npy", allow_pickle=True)
                FP_loss = np.load(f"{RESULTS_DIR}/losses/{args.dataset}/{args.method}/{args.task_type}/FP_loss_{i}_{j}.npy", allow_pickle=True)
                temp_lst.append(np.sum(NN_loss) + np.sum(FP_loss))
            score_lst.append(temp_lst)
        score_lst = np.array(score_lst)
        score_lst = score_lst.reshape(-1, args.n_repeats, 1)
else:
    score_lst = np.load(f"{RESULTS_DIR}/scores/{args.dataset}/{args.method}/{args.task_type}/score_list_{args.metric_type}.npy", allow_pickle=True)
    score_lst = score_lst.reshape(-1, args.n_repeats*args.n_repeats, 1)
    

                    
if args.metric_type == "Jaccard":
    x_axis = x_axis[1:]
    score_lst = 1-score_lst

# plot the score
plt.figure(figsize=(10, 6))
plt.title(f"Score for {args.dataset} with {args.method} and {args.task_type} task")
plt.xlabel("Label Weight")
plt.ylabel("Score")
# plot the mean and std
mean_score = np.mean(score_lst, axis=1)
std_score = np.std(score_lst, axis=1)

for i in range(mean_score.shape[1]):
    plt.plot(x_axis, mean_score[:, i], label=f"{args.method} {args.task_type} {args.metric_type} {i}")
    plt.fill_between(x_axis, mean_score[:, i] - std_score[:, i], mean_score[:, i] + std_score[:, i], alpha=0.2)
# plt.plot(x_axis, mean_score, label=f"{args.method} {args.task_type} {args.metric_type}")
# plt.fill_between(x_axis, mean_score - std_score, mean_score + std_score, alpha=0.2)
plt.xscale('log')
plt.legend()
plt.tight_layout()
os.makedirs(f"{RESULTS_DIR}/visualization/{args.dataset}/{args.method}/{args.task_type}", exist_ok=True)
plt.savefig(f"{RESULTS_DIR}/visualization/{args.dataset}/{args.method}/{args.task_type}/{args.metric_type}.png")
plt.close()

    
    