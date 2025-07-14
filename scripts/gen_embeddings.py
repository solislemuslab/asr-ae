import sys
import os 
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.decomposition import PCA
import torch
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilities.vae import load_model, load_data
from utilities.seq import aa_to_int_from_path
from utilities.utils import get_directory, parse_model_name, get_real_internals, one_hot_encode

def plot_embeddings(mu_leaves, mu_internal, data_path, model_name, 
                    valid_idx=None):
    """
    Plot the embeddings.
    """
    # combine embeddings of leaves and internal nodes
    mu = torch.cat((mu_leaves, mu_internal), dim=0)

    # coloring scheme: 
    labels = {
        "red": "Internal",
        "green": "Training tip",
        "orange": "Validation tip",
    }
    # Form a list that has colors in the right order
    if valid_idx is not None:
        col = ["orange" if i in valid_idx else "green" for i in range(len(mu_leaves))] 
    else:
        col = ["green" for _ in range(len(mu_leaves))]
    col += ["red" for _ in range(len(mu_internal))]
    
    # reduce dimension of embdeddings to 2D if it is higher with pca
    dim_embed = mu.shape[1]
    if dim_embed > 2:
        pca = PCA(n_components=2)
        mu = pca.fit_transform(mu)

    # plot the embeddings
    plt.figure(figsize=(5, 4))
    for color in labels.keys():
        mask = [c == color for c in col]
        plt.scatter(mu[mask, 0], mu[mask, 1], s=1, alpha=1, c=color, label=labels[color])
    if dim_embed > 2:
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
    else:
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
    plt.legend(loc="lower right", fontsize=8)
    plt.title('Embeddings of sequences')

    # save plot
    plot_dir = get_directory(data_path, "plots")
    plot_name = os.path.splitext(model_name)[0] + "_embeddings.png"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/{plot_name}", bbox_inches='tight')

def save_embeddings(mu_leaves, mu_internal, all_ids, data_path, model_name):
    """
    Save the embeddings to a csv file.
    """
    # combine embeddings of leaves and internal nodes
    mu = torch.cat((mu_leaves, mu_internal), dim=0).numpy()
    # Convert to dataframe
    embeddings = pd.DataFrame(mu, columns=[f"dim{i}" for i in range(mu.shape[1])])
    embeddings.insert(0, "id", all_ids)
    # Save to csv
    embeddings_name = os.path.splitext(model_name)[0] + "_embeddings.csv"
    embeddings_dir = get_directory(data_path, "embeddings", data_subfolder=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    embeddings.to_csv(f"{embeddings_dir}/{embeddings_name}", index=False)

def main():
    # Get config parameters
    parser = argparse.ArgumentParser(description='Generate embeddings of all ancestral and leaf sequences with trained VAE model.')
    parser.add_argument('config_file', nargs='?', default='config.json', 
                    help='Path to configuration file specifying details, such as which family reconstructions are for, etc.')
    args = parser.parse_args()
    print(f"Executing {sys.argv[0]} following {args.config_file}")
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    msa_path, data_path = config["msa_path"], config["data_path"]
    model_name = config["generate"]["model_name"]
    ding_model = model_name.startswith("ding")
    plot = config["generate"]["plot"]
    model_gapped_data_not = config["generate"]["model_gapped_data_not"]
    # Get model hyperparameters from name
    is_trans, ld, num_hidden_units, dim_aa_embed, one_hot = parse_model_name(model_name)
    # load leaf sequence data
    leaf_data, nl, nc = load_data(data_path, one_hot=one_hot)
    leaf_onehot, leaf_ids = leaf_data.msa, leaf_data.seq_keys
    # load internal sequence data
    aa_index = aa_to_int_from_path(data_path)
    with open(f"{data_path}/pos_preserved.pkl", 'rb') as file:
        pos_preserved = pickle.load(file)    
    internal_int, internal_ids  = get_real_internals(msa_path, aa_index, pos_preserved=pos_preserved)
    internal_onehot = one_hot_encode(internal_int, force_gap=model_gapped_data_not)
    internal_onehot= torch.tensor(internal_onehot).to(torch.float32) 
    # load model
    model_dir = get_directory(data_path, "saved_models")
    model_path = os.path.join(model_dir, model_name)
    model = load_model(model_path, nl=nl, nc=nc, ding_model=ding_model,
                       num_hidden_units=num_hidden_units, nlatent=ld,
                       one_hot=one_hot, dim_aa_embed=dim_aa_embed, trans=is_trans)  
    # get embeddings 
    with torch.no_grad():
        mu_leaves, _ = model.encoder(leaf_onehot)
        mu_internal, _ = model.encoder(internal_onehot)
    mu_leaves, mu_internal = mu_leaves.cpu(), mu_internal.cpu()
    # get indices of validation set
    with open(f"{model_dir}/valid_idx.pkl", 'rb') as file_handle:
        valid_idx = pickle.load(file_handle)
    # plot embeddings
    if plot:
        plot_embeddings(mu_leaves, mu_internal, data_path, model_name, 
                        valid_idx)
    # save embeddings
    all_ids = leaf_ids + internal_ids
    save_embeddings(mu_leaves, mu_internal, all_ids, data_path, model_name)

if __name__ == "__main__":
    main()