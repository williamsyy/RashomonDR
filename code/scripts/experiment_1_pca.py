# Visualize the PCA axis in 2D space
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data import data_prep
import argparse
import os
import torch

parser = argparse.ArgumentParser(description='PCA Visualization')
parser.add_argument('--data', type=str, default='MNIST', help='Dataset to use')
parser.add_argument('--missing_ratio', type=float, default=0.9, help='Missing ratio for labels')
parser.add_argument('--method', type=str, default='pacmap', help='Method to use for visualization')
parser.add_argument('--n_repeats', type=int, default=1, help='Number of repeats for the experiment')
parser.add_argument('--label_weights', type=float, default=10, help='Label weights for the experiment')
parser.add_argument('--plot_pca', action='store_true', help='Plot PCA visualization')

args = parser.parse_args()

# Results directory: set RESULTS_DIR environment variable or defaults to ./results
RESULTS_DIR = os.environ.get('RESULTS_DIR', './results')

# Load the dataset
X, y = data_prep(args.data)

# Perform PCA
first_pca = PCA(n_components=2)
X_pca = first_pca.fit_transform(X)

for random_seed in range(args.n_repeats):
    # load the embedding file
    if os.path.exists(f'{RESULTS_DIR}/embeddings/{args.data}/{args.method}/pca/{args.label_weights}_{args.missing_ratio}_{random_seed}.npy'):
        embedding = np.load(f'{RESULTS_DIR}/embeddings/{args.data}/{args.method}/pca/{args.label_weights}_{args.missing_ratio}_{random_seed}.npy')
        # load the model
        model = torch.load(f'{RESULTS_DIR}/models/{args.data}/{args.method}/pca/{args.label_weights}_{args.missing_ratio}_{random_seed}.pt', weights_only=False)
        if args.method != "pacmap":
            model = model[1]
        print(f'Embedding file already exists for dataset: {args.data}, random seed: {random_seed}.')
    else:
        print(f"{RESULTS_DIR}/embeddings/{args.data}/{args.method}/pca/{args.label_weights}_{args.missing_ratio}_{random_seed}.npy")
        print(f'Embedding file does not exist for dataset: {args.data}, random seed: {random_seed}.')
        continue
    if args.plot_pca:
        embedding = X_pca
        model = first_pca

    if  X.shape[1] > 100:
        projector = PCA(n_components=100)
        X_original = projector.fit_transform(X)
    
    # create the x axis representation
    X_pca_x_axis = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), num=10000)
    X_pca_x_axis_points = np.array([X_pca_x_axis, np.zeros(X_pca_x_axis.shape)]).T
    X_pca_y_axis = np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), num=10000)
    X_pca_y_axis_points = np.array([np.zeros(X_pca_y_axis.shape), X_pca_y_axis]).T

    # transform to the high dimensional space
    X_pca_x_axis_points = first_pca.inverse_transform(X_pca_x_axis_points)
    X_pca_y_axis_points = first_pca.inverse_transform(X_pca_y_axis_points)
    

    if X.shape[1] > 100 and args.method == "pacmap":
        # project back to the original space
        X_pca_x_axis_points = projector.transform(X_pca_x_axis_points)
        X_pca_y_axis_points = projector.transform(X_pca_y_axis_points)

        dataset = torch.tensor(X_original, dtype=torch.float32).cuda()

        # projection
        if args.plot_pca:
            X_embedding = model.transform(dataset.cpu().detach().numpy())
        else:
            X_embedding,_ = model(dataset)

        X_embedding_x_axis = torch.tensor(X_pca_x_axis_points, dtype=torch.float32).cuda()
        X_embedding_y_axis = torch.tensor(X_pca_y_axis_points, dtype=torch.float32).cuda()

        if args.plot_pca:
            X_embedding_x_axis = model.transform(X_embedding_x_axis)
            X_embedding_y_axis = model.transform(X_embedding_y_axis)
        else:
            X_embedding_x_axis,_ = model(X_embedding_x_axis)
            X_embedding_y_axis,_ = model(X_embedding_y_axis)

        X_embedding = X_embedding.cpu().detach().numpy()
        X_embedding_x_axis = X_embedding_x_axis.cpu().detach().numpy()
        X_embedding_y_axis = X_embedding_y_axis.cpu().detach().numpy()
    
    elif args.method == "pacmap":
        dataset = torch.tensor(X, dtype=torch.float32).cuda()

        # projection
        if args.plot_pca:
            X_embedding = model.transform(dataset.cpu().detach().numpy())
        else:
            X_embedding,_ = model(dataset)

        X_embedding_x_axis = torch.tensor(X_pca_x_axis_points, dtype=torch.float32).cuda()
        X_embedding_y_axis = torch.tensor(X_pca_y_axis_points, dtype=torch.float32).cuda()

        if args.plot_pca:
            X_embedding_x_axis = model.transform(X_embedding_x_axis.cpu().detach().numpy())
            X_embedding_y_axis = model.transform(X_embedding_y_axis.cpu().detach().numpy())
        else:
            X_embedding_x_axis,_ = model(X_embedding_x_axis)
            X_embedding_y_axis,_ = model(X_embedding_y_axis)

            X_embedding = X_embedding.cpu().detach().numpy()
            X_embedding_x_axis = X_embedding_x_axis.cpu().detach().numpy()
            X_embedding_y_axis = X_embedding_y_axis.cpu().detach().numpy()
    else:
        dataset = torch.tensor(X, dtype=torch.float32).cuda()

        # projection
        if args.plot_pca:
            X_embedding = model.transform(dataset.cpu().detach().numpy())
        else:
            X_embedding = model(dataset)

        X_embedding_x_axis = torch.tensor(X_pca_x_axis_points, dtype=torch.float32).cuda()
        X_embedding_y_axis = torch.tensor(X_pca_y_axis_points, dtype=torch.float32).cuda()

        if args.plot_pca:
            X_embedding_x_axis = model.transform(X_embedding_x_axis.cpu().detach().numpy())
            X_embedding_y_axis = model.transform(X_embedding_y_axis.cpu().detach().numpy())
        else:
            X_embedding_x_axis = model(X_embedding_x_axis)
            X_embedding_y_axis = model(X_embedding_y_axis)

            X_embedding = X_embedding.cpu().detach().numpy()
            X_embedding_x_axis = X_embedding_x_axis.cpu().detach().numpy()
            X_embedding_y_axis = X_embedding_y_axis.cpu().detach().numpy()

    plt.figure(figsize=(10, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y, s=1, cmap="Spectral")
    plt.plot(X_embedding_x_axis[:, 0], X_embedding_x_axis[:, 1], color='red', label='PC1 direction', lw=3)
    plt.annotate('', 
                xy=(X_embedding_x_axis[-1, 0], X_embedding_x_axis[-1, 1]), 
                xytext=(X_embedding_x_axis[-2, 0], X_embedding_x_axis[-2, 1]),
                arrowprops=dict(arrowstyle='->', color='red', lw=5))

    plt.plot(X_embedding_y_axis[:, 0], X_embedding_y_axis[:, 1], color='blue', label='PC2 direction',lw=3)
    plt.annotate('', 
                xy=(X_embedding_y_axis[-1, 0], X_embedding_y_axis[-1, 1]), 
                xytext=(X_embedding_y_axis[-2, 0], X_embedding_y_axis[-2, 1]),
                arrowprops=dict(arrowstyle='->', color='blue', lw=5))
    plt.title(f'PCA Visualization for {args.data} with {args.method} method')
    plt.xlabel('PaCMAP Axis 1')
    plt.ylabel('PaCMAP Axis 2')
    plt.colorbar()
    os.makedirs(f'{RESULTS_DIR}/figures/{args.data}/{args.method}/pca', exist_ok=True)
    if args.plot_pca:
        plt.savefig(f'{RESULTS_DIR}/figures/{args.data}/{args.method}/pca/pca_{args.label_weights}_{args.missing_ratio}_{random_seed}.png')
    else:
        plt.savefig(f'{RESULTS_DIR}/figures/{args.data}/{args.method}/pca/{args.label_weights}_{args.missing_ratio}_{random_seed}.png')
    # plt.savefig(f'{RESULTS_DIR}/figures/{args.data}/{args.method}/pca/{args.label_weights}_{args.missing_ratio}_{random_seed}.png')
    plt.close()