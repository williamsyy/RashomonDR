# Experiment 1 for differnet missing ratios visualizations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data import data_prep
import os
import argparse
import pacmap
from evaluation import soft_jaccard_similarity,faster_centroid_triplet_eval,random_triplet_eval,faster_knn_eval_series, faster_svm_eval,record_loss_pacmap, record_loss_umap, record_NCVis_loss, record_tSNE_loss, record_infonce_loss
import tqdm

# Results directory: set RESULTS_DIR environment variable or defaults to ./results
RESULTS_DIR = os.environ.get('RESULTS_DIR', './results')

parser = argparse.ArgumentParser(description='Visualize the results of experiment 1')
parser.add_argument('--dataset', type=str, default='MNIST', help='dataset name')
parser.add_argument('--n_repeats', type=int, default=5, help='number of repeats')
parser.add_argument('--method', type=str, default="pacmap", help='method name')
parser.add_argument('--task_type', type=str, default="axis", help='task type')
parser.add_argument('--missing_ratio', type=float, default=0.9, help='missing ratio')
parser.add_argument('--metric_type', type=str, default="Jaccard", help='metric type')

args = parser.parse_args()

X, y = data_prep(args.dataset)

if args.metric_type == "Jaccard":
    # if the metric is Jaccard, we need to load the large NN graph
    if os.path.exists(f'{RESULTS_DIR}/large_NN_graphs/{args.dataset}_large_NN_graph.npy'):
        large_NN_graph = np.load(f'{RESULTS_DIR}/large_NN_graphs/{args.dataset}_large_NN_graph.npy')
    else:
        import pacmap
        model = pacmap.PaCMAP(n_neighbors=50)
        model.fit(X, save_pairs=True)
        large_NN_graph = model.pair_neighbors
        os.makedirs(f'{RESULTS_DIR}/large_NN_graphs', exist_ok=True)
        np.save(f'{RESULTS_DIR}/large_NN_graphs/{args.dataset}_large_NN_graph.npy', large_NN_graph)

# load the pair_neighbors
if args.metric_type == "Loss":
    # load the neighbors
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


base_embedding_lst = []
score_lst = []

if args.metric_type == "TripletPCA" or args.metric_type == "RandomTripletPCA":
    if os.path.exists(f'{RESULTS_DIR}/embeddings/pca/{args.dataset}.npy'):
        embedding = np.load(f'{RESULTS_DIR}/embeddings/pca/{args.dataset}.npy')
        X_pca = embedding
        print("Loading the PCA embedding for dataset: ", args.dataset)
        print(X_pca)
    else:
        from sklearn.decomposition import PCA
        model = PCA()
        X_pca = model.fit_transform(X)
        os.makedirs(f'{RESULTS_DIR}/embeddings/pca/{args.dataset}', exist_ok=True)


for missing_ratio in tqdm.tqdm([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    base_embedding_lst = []
    score_lst = []
    for label_weight in tqdm.tqdm([0.0, 0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
                0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
                0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
                1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,
                20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,
                100.0]):
        if args.method != "pacmap":
            label_weight = round(label_weight * 10000, 1)
        for seed in range(args.n_repeats):
            # load the embedding
            if os.path.exists(f'{RESULTS_DIR}/embeddings/{args.dataset}/{args.method}/{args.task_type}/{label_weight}_{missing_ratio}_{seed}.npy'):
                embedding = np.load(f'{RESULTS_DIR}/embeddings/{args.dataset}/{args.method}/{args.task_type}/{label_weight}_{missing_ratio}_{seed}.npy')
                print("Loading the embedding for dataset: ", args.dataset, "with label weight: ", label_weight, "and random seed: ", seed)
            else:
                print("Skipping the embedding for dataset: ", args.dataset, "with label weight: ", label_weight, "and random seed: ", seed)
                continue
            
            if (label_weight == 0.0) and (args.metric_type == "Jaccard"):
                base_embedding_lst.append(embedding)
                continue

            if args.metric_type == "Jaccard":
                for embeddings in base_embedding_lst:
                    # compute the Jaccard distance
                    J, distance = soft_jaccard_similarity(embeddings, embedding, large_NN_graph)
                    score_lst.append(J)
                    
            elif args.metric_type == "TripletLoss":
                score = faster_centroid_triplet_eval(X,embedding,y)
                score_lst.append(score)

            elif args.metric_type == "RandomTripletLoss":
                score = random_triplet_eval(X,embedding,y)
                score_lst.append(score)
            
            elif args.metric_type == "FastKNN":
                score = faster_knn_eval_series(embedding,y)
                score_lst.append(score)
            
            elif args.metric_type == "Silhouette":
                from sklearn.metrics import silhouette_score
                score = silhouette_score(embedding, y)
                score_lst.append(score)
            
            elif args.metric_type == "TripletPCA":
                score = faster_centroid_triplet_eval(X_pca,embedding,y)
                score_lst.append(score)
                
            elif args.metric_type == "RandomTripletPCA":
                score = random_triplet_eval(X_pca,embedding,y)
                score_lst.append(score)
                
            elif args.metric_type == "SVMLoss":
                score = faster_svm_eval(embedding,y)
                score_lst.append(score)
            
            elif args.metric_type == "Loss":
                # load the pair neighbors
                pair_neighbors = np.load(f'{RESULTS_DIR}/generated_pairs/{args.dataset}_pair_neighbors_{seed}.npy')
                pair_MNs = np.load(f'{RESULTS_DIR}/generated_pairs/{args.dataset}_pair_MNs_{seed}.npy')
                pair_FPs = np.load(f'{RESULTS_DIR}/generated_pairs/{args.dataset}_pair_FPs_{seed}.npy')

                if args.method == "pacmap":
                    NN_loss, MN_loss, FP_loss = record_loss_pacmap(embedding, pair_neighbors.astype(np.int32), pair_MNs.astype(np.int32), pair_FPs.astype(np.int32), X.shape[0])
                elif args.method == "umap":
                    NN_loss, MN_loss, FP_loss = record_loss_umap(embedding, pair_neighbors.astype(np.int32), pair_MNs.astype(np.int32), pair_FPs.astype(np.int32), X.shape[0])
                elif args.method == "ncvis":
                    NN_loss, MN_loss, FP_loss = record_NCVis_loss(embedding, pair_neighbors.astype(np.int32), pair_MNs.astype(np.int32), pair_FPs.astype(np.int32), X.shape[0])
                elif args.method == "negtsne":
                    NN_loss, MN_loss, FP_loss = record_tSNE_loss(embedding, pair_neighbors.astype(np.int32), pair_MNs.astype(np.int32), pair_FPs.astype(np.int32), X.shape[0])
                elif args.method == "infonce":
                    NN_loss, MN_loss, FP_loss = record_infonce_loss(embedding, pair_neighbors.astype(np.int32), pair_MNs.astype(np.int32), pair_FPs.astype(np.int32), X.shape[0])
                
                if args.method == "pacmap":
                    # save the losses into three files
                    os.makedirs(f'{RESULTS_DIR}/losses/{args.dataset}/{args.method}/{args.task_type}/{missing_ratio}/', exist_ok=True)
                    np.save(f'{RESULTS_DIR}/losses/{args.dataset}/{args.method}/{args.task_type}/{missing_ratio}/NN_loss_{label_weight}_{seed}.npy', NN_loss)
                    np.save(f'{RESULTS_DIR}/losses/{args.dataset}/{args.method}/{args.task_type}/{missing_ratio}/MN_loss_{label_weight}_{seed}.npy', MN_loss)
                    np.save(f'{RESULTS_DIR}/losses/{args.dataset}/{args.method}/{args.task_type}/{missing_ratio}/FP_loss_{label_weight}_{seed}.npy', FP_loss)
                elif (args.method == "umap") or (args.method == "ncvis") or (args.method == "negtsne") or (args.method == "infonce"):
                    # save the losses into three files
                    os.makedirs(f'{RESULTS_DIR}/losses/{args.dataset}/{args.method}/{args.task_type}/{missing_ratio}', exist_ok=True)
                    np.save(f'{RESULTS_DIR}/losses/{args.dataset}/{args.method}/{args.task_type}/{missing_ratio}/NN_loss_{label_weight}_{seed}.npy', NN_loss)
                    np.save(f'{RESULTS_DIR}/losses/{args.dataset}/{args.method}/{args.task_type}/{missing_ratio}/FP_loss_{label_weight}_{seed}.npy', FP_loss)
                    

    if args.metric_type != "Loss":
        # check if the metric type is valid
        if args.metric_type not in ["Jaccard", "TripletLoss", "RandomTripletLoss", "FastKNN", "Silhouette", "TripletPCA", "SVMLoss", "RandomTripletPCA"]:
            raise ValueError("Invalid metric type. Please choose from Jaccard, TripletLoss, RandomTripletLoss, FastKNN, Silhouette, TripletPCA, SVMLoss.")
        score_lst = np.array(score_lst)
        # save the score list
        os.makedirs(f'{RESULTS_DIR}/scores/{args.dataset}/{args.method}/{args.task_type}/{missing_ratio}', exist_ok=True)
        np.save(f'{RESULTS_DIR}/scores/{args.dataset}/{args.method}/{args.task_type}/{missing_ratio}/score_list_{args.metric_type}.npy', score_lst)
        