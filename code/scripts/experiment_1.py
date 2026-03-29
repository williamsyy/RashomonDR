# Experiment 1 Generating the Embedding based on a selected label weights, missing ratio, and method
import argparse
import os
import random
import time
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
from data import data_prep
import pandas as pd
import torch
from parampacmap import parampacmap

# Results directory: set RESULTS_DIR environment variable or defaults to ./results
RESULTS_DIR = os.environ.get('RESULTS_DIR', './results')

parser = argparse.ArgumentParser(description='PaCMAP with classification loss')
parser.add_argument('--dataset', type=str, default='MNIST', help='dataset name')
parser.add_argument('--missing_ratio', type=float, default=0.9, help='missing ratio')
parser.add_argument('--label_weights', type=float, default=10, help='label weights')
parser.add_argument('--n_repeats', type=int, default=1, help='number of repeats')
parser.add_argument('--method', type=str, default="pacmap", help='method name')
parser.add_argument('--task_type', type=str, default="concept", help='task type')

args = parser.parse_args()

# load the dataset
X, y = data_prep(args.dataset)

# check if the pair indices exists
for random_seed in range(args.n_repeats):
    if not os.path.exists(f'{RESULTS_DIR}/generated_pairs/{args.dataset}_pair_neighbors_{random_seed}.npy'):
        import pacmap
        model = pacmap.PaCMAP(random_state=random_seed)
        _ = model.fit_transform(X, save_pairs=True)
        pair_neighbors = model.pair_neighbors
        pair_MNs = model.pair_MN
        pair_FPs = model.pair_FP
        # save the pair neighbors
        os.makedirs(f'{RESULTS_DIR}/generated_pairs', exist_ok=True)
        np.save(f'{RESULTS_DIR}/generated_pairs/{args.dataset}_pair_neighbors_{random_seed}.npy', pair_neighbors)
        np.save(f'{RESULTS_DIR}/generated_pairs/{args.dataset}_pair_MNs_{random_seed}.npy', pair_MNs)
        np.save(f'{RESULTS_DIR}/generated_pairs/{args.dataset}_pair_FPs_{random_seed}.npy', pair_FPs)
        print("Pair neighbors generated for dataset: ", args.dataset, "with random seed: ", random_seed)    

for random_seed in range(args.n_repeats):
    y_missing = y.copy()
    # random select the indices of the labels to be removed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random_indices = np.random.choice(len(y), int(args.missing_ratio * len(y)), replace=False)
    y_missing[random_indices] = -1
    # save the missing labels
    os.makedirs(f'{RESULTS_DIR}/missing_labels', exist_ok=True)
    np.save(f'{RESULTS_DIR}/missing_labels/{args.dataset}_{args.missing_ratio}_missing_labels_{random_seed}.npy', y_missing)

    if args.method == "pacmap":
        # check if the cuda is available
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        model_dict = {
            "backbone": "ANN",
            "layer_size": [100, 100, 100],
            "activation": "relu",
            "n_classes": len(np.unique(y)),
        }

        # apply PaCMAP to the data
        weight_schedule = parampacmap.pacmap_weight_schedule
        const_schedule = None
        weight_schedule = parampacmap.pacmap_weight_schedule
        loss_coeffs = [1, 1, 1]
        use_ns_loader = True
        activation = "relu"

        model = parampacmap.ParamPaCMAP(
            n_components=2,
            model_dict=model_dict,
            weight_schedule=weight_schedule,
            const_schedule=const_schedule,
            loss_coeffs=loss_coeffs,
            num_workers=12, # can be modified for the use case
            use_ns_loader=use_ns_loader,
            label_weight = args.label_weights,
            num_epochs=450,
            task_type=args.task_type,
            seed=random_seed,
        )

        # load the pair neighbors
        pair_neighbors = np.load(f'{RESULTS_DIR}/generated_pairs/{args.dataset}_pair_neighbors_{random_seed}.npy')
        pair_MNs = np.load(f'{RESULTS_DIR}/generated_pairs/{args.dataset}_pair_MNs_{random_seed}.npy')
        pair_FPs = np.load(f'{RESULTS_DIR}/generated_pairs/{args.dataset}_pair_FPs_{random_seed}.npy')
        model.pair_neighbors = pair_neighbors
        model.pair_MN = pair_MNs
        model.pair_FP = pair_FPs
        model._pairs_saved = True

    elif args.method == "ncvis":
        import cne
        model = cne.CNE(optimizer="adam", parametric=True, loss_mode="nce",label_weights=args.label_weights,seed=random_seed,task_type=args.task_type)
    elif args.method == "umap":
        import cne
        model = cne.CNE(optimizer="adam", parametric=True, loss_mode="umap",label_weights=args.label_weights,seed=random_seed,task_type=args.task_type)
    elif args.method == "negtsne":
        import cne
        model = cne.CNE(optimizer="adam", parametric=True, loss_mode="neg",label_weights=args.label_weights,seed=random_seed,task_type=args.task_type)
    elif args.method == "infonce":
        import cne
        model = cne.CNE(optimizer="adam", parametric=True, loss_mode="infonce",label_weights=args.label_weights,seed=random_seed,task_type=args.task_type)
    else:
        raise ValueError("method should be either 'pacmap', 'ncvis', 'umap', 'negtsne' or 'infonce'")

    if args.task_type == "pca":
        # apply pca into the model
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca= pca.fit_transform(X)
        X_embedding = model.fit_transform(X.astype(np.float32), X_pca.astype(np.float32))
    elif args.task_type == "concept":
        # apply axis into the model
        X_embedding = model.fit_transform(X.astype(np.float32), y_missing.astype(np.int64))
    else:
        raise ValueError("task_type should be either 'pca' or 'concept")
    
    

        
    # save the embedding
    os.makedirs(f'{RESULTS_DIR}/embeddings', exist_ok=True)
    # create a folder for the dataset
    os.makedirs(f'{RESULTS_DIR}/embeddings/{args.dataset}', exist_ok=True)
    # create a folder for the method
    os.makedirs(f'{RESULTS_DIR}/embeddings/{args.dataset}/{args.method}', exist_ok=True)
    # create a folder for the task type
    os.makedirs(f'{RESULTS_DIR}/embeddings/{args.dataset}/{args.method}/{args.task_type}', exist_ok=True)
    np.save(f'{RESULTS_DIR}/embeddings/{args.dataset}/{args.method}/{args.task_type}/{args.label_weights}_{args.missing_ratio}_{random_seed}.npy', X_embedding)
    # plot the embedding
    plt.figure(figsize=(10, 10))
    plt.scatter(X_embedding[:, 0], X_embedding[:, 1], c=y, cmap='Spectral', s=1)
    plt.title(f'PaCMAP embedding of {args.dataset} dataset with random_state={random_seed}, label_weight={args.label_weights}')
    plt.colorbar()
    os.makedirs(f'{RESULTS_DIR}/plots', exist_ok=True)
    os.makedirs(f'{RESULTS_DIR}/plots/{args.dataset}', exist_ok=True)
    os.makedirs(f'{RESULTS_DIR}/plots/{args.dataset}/{args.method}', exist_ok=True)
    os.makedirs(f'{RESULTS_DIR}/plots/{args.dataset}/{args.method}/{args.task_type}', exist_ok=True)
    plt.savefig(f'{RESULTS_DIR}/plots/{args.dataset}/{args.method}/{args.task_type}/{args.label_weights}_{args.missing_ratio}_{random_seed}.png')
    plt.close()

    if args.method == "pacmap":
        os.makedirs(f'{RESULTS_DIR}/models/', exist_ok=True)
        os.makedirs(f'{RESULTS_DIR}/models/{args.dataset}', exist_ok=True)
        os.makedirs(f'{RESULTS_DIR}/models/{args.dataset}/{args.method}', exist_ok=True)
        os.makedirs(f'{RESULTS_DIR}/models/{args.dataset}/{args.method}/{args.task_type}', exist_ok=True)
        torch.save(model.model, f'{RESULTS_DIR}/models/{args.dataset}/{args.method}/{args.task_type}/{args.label_weights}_{args.missing_ratio}_{random_seed}.pt')
    else:
        os.makedirs(f'{RESULTS_DIR}/models/', exist_ok=True)
        os.makedirs(f'{RESULTS_DIR}/models/{args.dataset}', exist_ok=True)
        os.makedirs(f'{RESULTS_DIR}/models/{args.dataset}/{args.method}', exist_ok=True)
        os.makedirs(f'{RESULTS_DIR}/models/{args.dataset}/{args.method}/{args.task_type}', exist_ok=True)
        torch.save(model.model, f'{RESULTS_DIR}/models/{args.dataset}/{args.method}/{args.task_type}/{args.label_weights}_{args.missing_ratio}_{random_seed}.pt')


