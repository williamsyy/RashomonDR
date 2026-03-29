# Experiment 2: Generate common knowledge embeddings using the large NN graph and a given cutoff label weight.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data import data_prep
import os
import argparse
from parampacmap import parampacmap
from parampacmap import utils
import pacmap
import torch


parser = argparse.ArgumentParser(description='Visualize the results of experiment 2')
parser.add_argument('--dataset', type=str, default='MNIST', help='dataset name')
parser.add_argument('--method', type=str, default="pacmap", help='method name')
parser.add_argument('--task_type', type=str, default="concept", help='task type')
parser.add_argument('--missing_ratio', type=float, default=0.9, help='missing ratio')
parser.add_argument('--label_weights_cutoff', type=float, default=0.01, help='label weights cutoff')
parser.add_argument('--n_repeats', type=int, default=5, help='number of repeats')

args = parser.parse_args()

# Results directory: set RESULTS_DIR environment variable or defaults to ./results
RESULTS_DIR = os.environ.get('RESULTS_DIR', './results')

X, y = data_prep(args.dataset)
if (args.dataset == "FMNIST"):
    # 0 Ankle boot 1 Sneaker 2 Sandal 3 Trouser 4 Dress 5 Pullover 6 Coat 7 T-shirt/top 8 Shirt 9 Bag
    mapping = np.array([7, 3, 5, 4, 6, 2, 8, 1, 9, 0])
    y = mapping[y]
# check if there exists a large NN graph for the dataset
for random_seed in range(args.n_repeats):
    if os.path.exists(f'{RESULTS_DIR}/large_NN_graphs/{args.dataset}_large_NN_graph_{random_seed}.npy'):
        large_NN_graph = np.load(f'{RESULTS_DIR}/large_NN_graphs/{args.dataset}_large_NN_graph_{random_seed}.npy')
    else:
        import pacmap
        model = pacmap.PaCMAP(n_neighbors=50, random_state=random_seed)
        model.fit(X, save_pairs=True)
        large_NN_graph = model.pair_neighbors
        os.makedirs(f'{RESULTS_DIR}/large_NN_graphs', exist_ok=True)
        np.save(f'{RESULTS_DIR}/large_NN_graphs/{args.dataset}_large_NN_graph_{random_seed}.npy', large_NN_graph)

def calculate_NN_distance(embedding, large_NN_graph):
    # Get the embeddings for the start and end nodes
    start_embeds = embedding[large_NN_graph[:, 0]]  # shape (n_edges, d)
    end_embeds = embedding[large_NN_graph[:, 1]]    # shape (n_edges, d)

    # Compute pairwise differences
    diffs = start_embeds - end_embeds  # shape (n_edges, d)

    # Compute Euclidean norm (vectorized)
    distance_NN = np.linalg.norm(diffs, axis=1)  # shape (n_edges,)

    return distance_NN
for random_seed in range(args.n_repeats):
    current_score_NN = []
    distance_NN_lst = []
    for file in os.listdir(f'{RESULTS_DIR}/embeddings/{args.dataset}/{args.method}/{args.task_type}'):
        label_weights = float(file.split('_')[0])
        if label_weights > args.label_weights_cutoff:
            continue

        # load the embedding
        embedding = np.load(f'{RESULTS_DIR}/embeddings/{args.dataset}/{args.method}/{args.task_type}/{file}')

        # compute the distance of the nearest neighbors
        distance_NN = calculate_NN_distance(embedding, large_NN_graph)

        distance_NN_lst.append(distance_NN)

    distance_NN_arr = np.array(distance_NN_lst)

    def robust_std(column):
        med = np.median(column)
        mad = np.median(np.abs(column - med))
        return 1.4826 * mad

    variance_distance_NN = np.apply_along_axis(robust_std, axis=0, arr=distance_NN_lst)
    max_distance_NN = np.max(distance_NN_arr, axis=0)
    median_distance_NN = np.median(distance_NN_arr, axis=0)
    global_median_distance_NN = np.median(distance_NN_arr)
    select_distance= np.where(median_distance_NN > global_median_distance_NN, median_distance_NN, max_distance_NN)
    # normalized the variance and max
    variance_distance_NN = (variance_distance_NN - np.min(variance_distance_NN)) / (np.max(variance_distance_NN) - np.min(variance_distance_NN)+1e-10)
    select_distance = (select_distance - np.min(select_distance)) / (np.max(select_distance) - np.min(select_distance)+1e-10)
    distance_score_NN = variance_distance_NN + select_distance

    distance_score_NN = pd.DataFrame(distance_score_NN, columns=['score'])
    distance_score_NN[["x","y"]] = large_NN_graph
    # df_unique = distance_score_NN.sort_values('score', ascending=True)
    if args.method == "pacmap":
        df_top = distance_score_NN.groupby('x', group_keys=False).apply(lambda g: g.nsmallest(10, 'score'))
        top_arr = df_top[['x', 'y']].to_numpy()
    else:
        df_top = distance_score_NN.groupby('x', group_keys=False).apply(lambda g: g.nsmallest(30, 'score'))
        top_arr = df_top[['x', 'y']].to_numpy()

    if args.method == "pacmap":
        # refit the model with the top pairs
        weight_schedule = parampacmap.pacmap_weight_schedule
        const_schedule = None
        loss_coeffs = [1, 1, 1]
        use_ns_loader = True
        activation = "relu"

        import torch
        # check if cuda is available
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        model_dict = {
            "backbone": "ANN",
            "layer_size": [100, 100, 100],
            "activation": activation,
            "n_classes": len(np.unique(y)),
        }

        model = parampacmap.ParamPaCMAP(
            n_components=2,
            model_dict=model_dict,
            weight_schedule=weight_schedule,
            const_schedule=const_schedule,
            loss_coeffs=loss_coeffs,
            num_workers=12,
            use_ns_loader=use_ns_loader,
            label_weight = 0,
            seed = random_seed,
            task_type=args.task_type,
        )

        pair_neighbors, pair_MN, pair_FP, _ = utils.data.generate_pair(X.astype(np.float32), n_neighbors=10, n_MN=5, n_FP=20, pair_neighbors=top_arr)

        model.pair_neighbors = top_arr
        model.pair_MN = pair_MN
        model.pair_FP = pair_FP
        model._pairs_saved = True

        if args.task_type == "concept":
            X_embedding = model.fit_transform(X.astype(np.float32), y.astype(np.int64))
        elif args.task_type == "pca":
            X_embedding = model.fit_transform(X.astype(np.float32), np.zeros((X.shape[0],2), dtype=np.int64))
    elif args.method == "umap":
        from scipy.sparse import csr_matrix
        data = np.ones(len(top_arr))
        n_points = np.max(top_arr) + 1
        adj_matrix = csr_matrix((data, (top_arr[:, 0], top_arr[:, 1])), shape=(n_points, n_points))
        import cne
        model = cne.CNE(optimizer="adam", parametric=True, loss_mode="umap",label_weights=0,seed=random_seed,task_type="")
        X_embedding = model.fit_transform(X.astype(np.float32), y.astype(np.int64),graph=adj_matrix)
    elif args.method == "negtsne":
        from scipy.sparse import csr_matrix
        data = np.ones(len(top_arr))
        n_points = np.max(top_arr) + 1
        adj_matrix = csr_matrix((data, (top_arr[:, 0], top_arr[:, 1])), shape=(n_points, n_points))
        import cne
        model = cne.CNE(optimizer="adam", parametric=True, loss_mode="neg",label_weights=0,seed=random_seed,task_type="")
        X_embedding = model.fit_transform(X.astype(np.float32), y.astype(np.int64),graph=adj_matrix)
    elif args.method == "infonce":
        from scipy.sparse import csr_matrix
        data = np.ones(len(top_arr))
        n_points = np.max(top_arr) + 1
        adj_matrix = csr_matrix((data, (top_arr[:, 0], top_arr[:, 1])), shape=(n_points, n_points))
        import cne
        model = cne.CNE(optimizer="adam", parametric=True, loss_mode="infonce",label_weights=0,seed=random_seed,task_type="")
        X_embedding = model.fit_transform(X.astype(np.float32), y.astype(np.int64),graph=adj_matrix)
    elif args.method == "ncvis":
        from scipy.sparse import csr_matrix
        data = np.ones(len(top_arr))
        n_points = np.max(top_arr) + 1
        adj_matrix = csr_matrix((data, (top_arr[:, 0], top_arr[:, 1])), shape=(n_points, n_points))
        import cne
        model = cne.CNE(optimizer="adam", parametric=True, loss_mode="nce",label_weights=0,seed=random_seed,task_type="")
        X_embedding = model.fit_transform(X.astype(np.float32), y.astype(np.int64),graph=adj_matrix)

    # save the embedding
    os.makedirs(f'{RESULTS_DIR}/common_knowledge_embeddings', exist_ok=True)
    np.save(f'{RESULTS_DIR}/common_knowledge_embeddings/{args.dataset}_{args.method}_{args.task_type}_{args.missing_ratio}_{random_seed}.npy', X_embedding)

    # save the distance score
    os.makedirs(f'{RESULTS_DIR}/common_knowledge_embeddings_distance_score', exist_ok=True)
    pd.DataFrame(distance_score_NN).to_csv(f'{RESULTS_DIR}/common_knowledge_embeddings_distance_score/{args.dataset}_{args.method}_{args.task_type}_{args.missing_ratio}_{random_seed}.csv', index=False)

    # plot the embedding
    plt.figure(figsize=(10, 10))
    plt.scatter(X_embedding[:, 0], X_embedding[:, 1], c=y, cmap='Spectral', s=1)
    plt.title(f'{args.dataset} dataset with label weight less or equal to {args.label_weights_cutoff}')
    os.makedirs(f'{RESULTS_DIR}/common_knowledge_embeddings_plots', exist_ok=True)
    plt.savefig(f'{RESULTS_DIR}/common_knowledge_embeddings_plots/{args.dataset}_{args.method}_{args.task_type}_{args.missing_ratio}_{random_seed}.png')
    plt.close()

    # save the model
    os.makedirs(f'{RESULTS_DIR}/common_knowledge_embeddings_models', exist_ok=True)
    torch.save(model.model, f'{RESULTS_DIR}/common_knowledge_embeddings_models/{args.dataset}_{args.method}_{args.task_type}_{args.missing_ratio}_{random_seed}.pth')