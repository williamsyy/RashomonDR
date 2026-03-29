# compare the results of experiment 2 with the initial embedding
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import argparse

# load the dataset
argparse = argparse.ArgumentParser(description='Visualize the results of experiment 2')
argparse.add_argument('--dataset', type=str, default='MNIST', help='dataset name')
argparse.add_argument('--task_type', type=str, default="axis", help='task type')
argparse.add_argument('--n_repeats', type=int, default=5, help='number of repeats')

args = argparse.parse_args()

# Results directory: set RESULTS_DIR environment variable or defaults to ./results
RESULTS_DIR = os.environ.get('RESULTS_DIR', './results')

methods_list = ["infonce", "ncvis", "negtsne", "umap", "pacmap"]

x_axis = [0.0, 0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
            0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
            0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
            1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,
            20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,
            100.0]
x_axis = np.array(x_axis)

# load the cutoff file
cutoff_df = pd.read_csv("cutoff.csv")

# generate a barplot for the difference between the loss and the cutoff values

methods = []
knn_means, knn_stds = [], []
jaccard_means, jaccard_stds = [], []
triplet_means, triplet_stds = [], []
random_triplet_means, random_triplet_stds = [], []
silhouette_means, silhouette_stds = [], []
loss_means, loss_stds = [], []
tripletpca_means, tripletpca_stds = [], []
randomtripletpca_means, randomtripletpca_stds = [], []
svm_means, svm_stds = [], []

for method in methods_list:
    if method == "pacmap":
        x_axis = [0.0, 0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
            0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
            0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
            1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,
            20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,
            100.0]
        x_axis = np.array(x_axis)
    else:
        x_axis = [0.0, 0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
            0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
            0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
            1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,
            20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,
            100.0]
        x_axis = np.array(x_axis)
        x_axis = (x_axis * 10000).astype(int).astype(float)
    # load the knn score
    knn_score = np.load(f"{RESULTS_DIR}/scores/{args.dataset}/{method}/{args.task_type}/score_list_FastKNN.npy", allow_pickle=True)
    knn_score = knn_score.reshape(-1, args.n_repeats, knn_score.shape[1])[:, :, 2]
    # select the 
    knn_score_mean = np.mean(knn_score, axis=1)
    knn_score_std = np.std(knn_score, axis=1)

    random_triplet_score = np.load(f"{RESULTS_DIR}/scores/{args.dataset}/{method}/{args.task_type}/score_list_RandomTripletLoss.npy", allow_pickle=True)
    random_triplet_score = random_triplet_score.reshape(-1, args.n_repeats)
    random_triplet_score_mean = np.mean(random_triplet_score, axis=1)
    random_triplet_score_std = np.std(random_triplet_score, axis=1)

    triplet_score = np.load(f"{RESULTS_DIR}/scores/{args.dataset}/{method}/{args.task_type}/score_list_TripletLoss.npy", allow_pickle=True)
    triplet_score = triplet_score.reshape(-1, args.n_repeats)
    triplet_score_mean = np.mean(triplet_score, axis=1)
    triplet_score_std = np.std(triplet_score, axis=1)

    sihouette_score = np.load(f"{RESULTS_DIR}/scores/{args.dataset}/{method}/{args.task_type}/score_list_Silhouette.npy", allow_pickle=True)
    sihouette_score = sihouette_score.reshape(-1, args.n_repeats)
    sihouette_score_mean = np.mean(sihouette_score, axis=1)
    sihouette_score_std = np.std(sihouette_score, axis=1)

    svm_score = np.load(f"{RESULTS_DIR}/scores/{args.dataset}/{method}/{args.task_type}/score_list_SVMLoss_0.0.npy", allow_pickle=True)
    svm_score = svm_score.reshape(-1, args.n_repeats)
    svm_score_mean = np.mean(svm_score, axis=1)
    svm_score_std = np.std(svm_score, axis=1)

    # get the first value
    silhouette_score_mean_selected = sihouette_score_mean[0]
    silhouette_score_std_selected = sihouette_score_std[0]
    knn_score_mean_selected = knn_score_mean[0]
    knn_score_std_selected = knn_score_std[0]
    random_triplet_score_mean_selected = random_triplet_score_mean[0]
    random_triplet_score_std_selected = random_triplet_score_std[0]
    triplet_score_mean_selected = triplet_score_mean[0]
    triplet_score_std_selected = triplet_score_std[0]
    svm_score_mean_selected = svm_score_mean[0]
    svm_score_std_selected = svm_score_std[0]
    # load the knn score
    knn_score = np.load(f"{RESULTS_DIR}/common_knowledge_embeddings_score/{args.dataset}_{method}_{args.task_type}_FastKNN.npy",allow_pickle=True)
    knn_score = knn_score.reshape(-1, 5, knn_score.shape[1])[:, :, 2]
    # select the 
    knn_score_mean = np.mean(knn_score)
    knn_score_std = np.std(knn_score)


    random_triplet_score = np.load(f"{RESULTS_DIR}/common_knowledge_embeddings_score/{args.dataset}_{method}_{args.task_type}_RandomTripletLoss.npy", allow_pickle=True)
    random_triplet_score = random_triplet_score.reshape(-1, 5)
    random_triplet_score_mean = np.mean(random_triplet_score)
    random_triplet_score_std = np.std(random_triplet_score)

    triplet_score = np.load(f"{RESULTS_DIR}/common_knowledge_embeddings_score/{args.dataset}_{method}_{args.task_type}_TripletLoss.npy", allow_pickle=True)
    triplet_score = triplet_score.reshape(-1, 5)
    triplet_score_mean = np.mean(triplet_score)
    triplet_score_std = np.std(triplet_score)

    sihouette_score = np.load(f"{RESULTS_DIR}/common_knowledge_embeddings_score/{args.dataset}_{method}_{args.task_type}_Silhouette.npy", allow_pickle=True)
    sihouette_score = sihouette_score.reshape(-1, 5)
    sihouette_score_mean = np.mean(sihouette_score)
    sihouette_score_std = np.std(sihouette_score)

    svm_score = np.load(f"{RESULTS_DIR}/common_knowledge_embeddings_score/{args.dataset}_{method}_{args.task_type}_SVMLoss.npy", allow_pickle=True)
    svm_score = svm_score.reshape(-1, 5)
    svm_score_mean = np.mean(svm_score)
    svm_score_std = np.std(svm_score)

    tripletpca_score = np.load(f"{RESULTS_DIR}/common_knowledge_embeddings_score/{args.dataset}_{method}_{args.task_type}_TripletPCA.npy", allow_pickle=True)
    tripletpca_score = tripletpca_score.reshape(-1, args.n_repeats)
    tripletpca_score_mean = np.mean(tripletpca_score, axis=1)
    tripletpca_score_std = np.std(tripletpca_score, axis=1)

    randomtripletpca_score = np.load(f"{RESULTS_DIR}/common_knowledge_embeddings_score/{args.dataset}_{method}_{args.task_type}_RandomTripletPCA.npy", allow_pickle=True)
    randomtripletpca_score = randomtripletpca_score.reshape(-1, args.n_repeats)
    randomtripletpca_score_mean = np.mean(randomtripletpca_score, axis=1)
    randomtripletpca_score_std = np.std(randomtripletpca_score, axis=1)


    methods.append(method)

    # KNN scores (all vs selected)
    knn_means.append([knn_score_mean_selected, knn_score_mean])
    knn_stds.append([knn_score_std_selected, knn_score_std])
    # Triplet satisfaction scores (all vs selected)
    triplet_means.append([triplet_score_mean_selected, triplet_score_mean])
    triplet_stds.append([triplet_score_std_selected, triplet_score_std])

    # Random triplet scores (all vs selected)
    random_triplet_means.append([random_triplet_score_mean_selected, random_triplet_score_mean])
    random_triplet_stds.append([random_triplet_score_std_selected, random_triplet_score_std])

    # Silhouette scores (all vs selected)
    silhouette_means.append([silhouette_score_mean_selected, sihouette_score_mean])
    silhouette_stds.append([silhouette_score_std_selected, sihouette_score_std])

    svm_means.append([svm_score_mean_selected,svm_score_mean])
    svm_stds.append([svm_score_std_selected,svm_score_std])
import matplotlib.pyplot as plt
import numpy as np

n_methods = len(methods)
x = np.arange(n_methods)  # the method index
bar_width = 0.35

# Unpack into arrays
knn_means = np.array(knn_means)
knn_stds = np.array(knn_stds)
jaccard_means = np.array(jaccard_means)
jaccard_stds = np.array(jaccard_stds)
triplet_means = np.array(triplet_means)
triplet_stds = np.array(triplet_stds)
random_triplet_means = np.array(random_triplet_means)
random_triplet_stds = np.array(random_triplet_stds)
silhouette_means = np.array(silhouette_means)
silhouette_stds = np.array(silhouette_stds)
loss_means = np.array(loss_means)
loss_stds = np.array(loss_stds)
tripletpca_means = np.array(tripletpca_means)
tripletpca_stds = np.array(tripletpca_stds)
randomtripletpca_means = np.array(randomtripletpca_means)
randomtripletpca_stds = np.array(randomtripletpca_stds)
svm_means = np.array(svm_means)
svm_stds = np.array(svm_stds)

metrics = {
    "5-NN Accuracy⬆": (knn_means, knn_stds),
    "Jaccard distance": (jaccard_means, jaccard_stds),
    "Triplet Score": (triplet_means, triplet_stds),
    "Random Triplet Score": (random_triplet_means, random_triplet_stds),
    "Silhouette Score⬆": (silhouette_means, silhouette_stds),
    "SVM Score⬆": (svm_means,svm_stds),
    "Triplet PCA Score": (tripletpca_means, tripletpca_stds),
    "Random Triplet PCA Score": (randomtripletpca_means, randomtripletpca_stds)
}


import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

import matplotlib.ticker as ticker

plt.rcParams.update({
    'axes.titlesize':24,       # Subplot titles
    'axes.labelsize':24,       # Axis labels
    'xtick.labelsize': 20,      # X tick labels
    'ytick.labelsize': 20,      # Y tick labels
    'legend.fontsize': 24,      # Legend font
    'figure.titlesize': 24      # Suptitle
})

fig, axs = plt.subplots(1, len(metrics), figsize=(15, 4), sharey=False)
if len(metrics) == 1:
    axs = [axs]
colors = plt.cm.tab10.colors
bar_width = 0.35
x = np.arange(len(methods))

for ax, (metric_name, (means, stds)) in zip(axs, metrics.items()):
    for i, method in enumerate(methods):
        color = colors[i % len(colors)]

        # Bar positions
        x_start = x[i] - bar_width / 2
        x_cutoff = x[i] + bar_width / 2

        # Start (0) — plain white fill, solid colored border
        ax.bar(x_start, means[i, 0],
               width=bar_width,
               color='white',
               edgecolor=color,
               linewidth=2,
               hatch='',
               label='Start (0)' if i == 0 and metric_name == list(metrics.keys())[0] else '')

        # Cutoff — same, but with hatch
        ax.bar(x_cutoff, means[i, 1],
               width=bar_width,
               color='white',
               edgecolor=color,
               linewidth=2,
               hatch='//',
               label='Cutoff' if i == 0 and metric_name == list(metrics.keys())[0] else '')

        # Add error bars manually (centered)
        ax.errorbar(x_start, means[i, 0], yerr=stds[i, 0],
                    fmt='none', ecolor=color, capsize=3)
        ax.errorbar(x_cutoff, means[i, 1], yerr=stds[i, 1],
                    fmt='none', ecolor=color, capsize=3)

    ax.axhline(0, color='gray', linewidth=1, linestyle='--')  # baseline
    ax.set_title(metric_name)
    # remove the xticks and make it disappear
    ax.set_xticks([])

    # Auto-range to include both positive and negative values
    ymin = np.min(means - stds)
    ymax = np.max(means + stds)
    if ymax > 1e4:
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(12)  # Optional: control the offset font size
    ax.set_ylim(ymin - 0.1 * abs(ymin), ymax + 0.1 * abs(ymax))

# Method color legend (border only)
method_legend = [
    Patch(facecolor='white', edgecolor=colors[i % len(colors)], linewidth=2, label=method)
    for i, method in enumerate(methods)
]

# Style legend: hatch = Cutoff, solid = Start
style_legend = [
    Patch(facecolor='white', edgecolor='black', linewidth=2, hatch='', label='Original DR'),
    Patch(facecolor='white', edgecolor='black', linewidth=2, hatch='//', label='Combined DR')
]

# === Add to LAST AXES ===
axs[-1].legend(handles=method_legend + style_legend,
               loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"metric_comparison_{args.dataset}_{args.task_type}_new.png",dpi=400)



    