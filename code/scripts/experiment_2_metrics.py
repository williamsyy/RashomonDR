import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data import data_prep
import os
import argparse
from evaluation import faster_centroid_triplet_eval,random_triplet_eval,faster_knn_eval_series, faster_svm_eval,record_loss_pacmap, record_loss_umap, record_NCVis_loss, record_tSNE_loss, record_infonce_loss,soft_jaccard_similarity

parser = argparse.ArgumentParser(description='Visualize the results of experiment 2')
parser.add_argument('--dataset', type=str, default='MNIST', help='dataset name')
parser.add_argument('--method', type=str, default="pacmap", help='method name')
parser.add_argument('--task_type', type=str, default="axis", help='task type')
parser.add_argument('--metric_type', type=str, default="Jaccard", help='metric type')
parser.add_argument('--n_repeats', type=int, default=5, help='number of repeats')

args = parser.parse_args()

# Results directory: set RESULTS_DIR environment variable or defaults to ./results
RESULTS_DIR = os.environ.get('RESULTS_DIR', './results')

X, y = data_prep(args.dataset)

score_lst = []

if (args.metric_type == "TripletPCA") or (args.metric_type == "RandomTripletPCA"):
    from sklearn.decomposition import PCA
    model = PCA()
    X_pca = model.fit_transform(X)

for seed in range(args.n_repeats):
    embedding = np.load(f'{RESULTS_DIR}/common_knowledge_embeddings/{args.dataset}_{args.method}_{args.task_type}_0.9_{seed}.npy')
    # standardize the embedding
    embedding = (embedding - np.mean(embedding, axis=0)) / np.std(embedding, axis=0)

    if args.metric_type == "TripletLoss":
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
    
    else:
        raise ValueError("Invalid metric type. Choose from 'TripletLoss', 'RandomTripletLoss', 'FastKNN', 'Silhouette', 'TripletPCA', 'RandomTripletPCA', or 'SVMLoss'.")

# save the score
os.makedirs(f'{RESULTS_DIR}/common_knowledge_embeddings_score', exist_ok=True)
np.save(f'{RESULTS_DIR}/common_knowledge_embeddings_score/{args.dataset}_{args.method}_{args.task_type}_{args.metric_type}.npy', score_lst)

