import sys
import os 
import re
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.decomposition import PCA
import torch
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilities.vae import load_model, load_data
from utilities.utils import get_directory, parse_model_name

def plot_embeddings(mu, data_path, model_name, valid_idx=None):
    """
    Plot the embeddings.
    """
    # title 
    
    # color points based on whether they are in the validation set
    if valid_idx:
        col = ["green" if i in valid_idx else "orange" for i in range(len(mu))]
    else:
        col = "orange"
    # reduce dimension of embdeddings to 2D if it is higher with pca
    dim_embed = mu.shape[1]
    if dim_embed > 2:
        pca = PCA(n_components=2)
        mu = pca.fit_transform(mu)
    plt.figure(figsize=(5, 4))
    plt.scatter(mu[:, 0], mu[:, 1], s=1, alpha=0.9, c = col)
    if valid_idx:
        plt.scatter([], [], c='green', label='Test', s=10)  # Dummy scatter for legend
        plt.scatter([], [], c='orange', label='Train', s=10)  # Dummy scatter for legend
    if dim_embed > 2:
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
    else:
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
    plt.title('Embeddings of sequences at tips of the tree')
    # save plot
    plot_dir = get_directory(data_path, "plots")
    plot_name = os.path.splitext(model_name)[0] + "_embeddings.png"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/{plot_name}", bbox_inches='tight')

def save_embeddings(mu, data_path, model_name):
    # Convert to dataframe
    embeddings = pd.DataFrame(mu.numpy(), columns=[f"dim{i}" for i in range(mu.shape[1])])
    # Add names of sequences as a column
    with open(f"{data_path}/seq_names.pkl", 'rb') as file_handle:
        seq_names = pickle.load(file_handle)
    embeddings.insert(0, "id", seq_names)
    # Save to csv
    embeddings_name = os.path.splitext(model_name)[0] + "_embeddings.csv"
    embeddings_dir = get_directory(data_path, "embeddings", data_subfolder=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    embeddings.to_csv(f"{embeddings_dir}/{embeddings_name}", index=False)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate embeddings for sequences at the tips of the tree.")
    parser.add_argument("data_path", type=str, help="Path to the data directory.")
    parser.add_argument("model_name", type=str, help="Name of the model file.")
    parser.add_argument("--plot", type=bool, default=True, help="Whether to plot the embeddings (default: False).")

    # Parse arguments
    args = parser.parse_args()
    data_path = args.data_path
    model_name = args.model_name
    plot = args.plot
    model_dir = get_directory(data_path, "saved_models")
    model_path = os.path.join(model_dir, model_name)
    ld, num_hidden_units, dim_aa_embed, one_hot = parse_model_name(model_name)
    # load data
    data, nl, nc = load_data(data_path, one_hot=one_hot)
    # load model
    model = load_model(model_path, nl=nl, nc=nc,
                           num_hidden_units=num_hidden_units, nlatent=ld,
                           one_hot=one_hot, dim_aa_embed=dim_aa_embed)  
    # get embeddings
    with torch.no_grad():
        mu, _ = model.encoder(data.msa)
    # get indices of validation set
    with open(f"{model_dir}/valid_idx.pkl", 'rb') as file_handle:
        valid_idx = pickle.load(file_handle)
    # plot embeddings
    if plot:
        plot_embeddings(mu, data_path, model_name, valid_idx)
    # save embeddings
    save_embeddings(mu, data_path, model_name)

if __name__ == "__main__":
    main()