import sys
import os 
import re
import json
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from autoencoder.modules.model import load_model
from utilities.utils import get_directory

def load_binary_data(data_path):
    """
    Load the data from the data path.
    """
    with open(f"{data_path}/seq_msa_binary.pkl", 'rb') as file_handle:
        msa_binary = torch.tensor(pickle.load(file_handle))
    # for some reason, msa_binary is float64 so first transform it to float 32 for later use
    msa_binary = msa_binary.to(torch.float32)
    nl = msa_binary.shape[1]
    nc = msa_binary.shape[2]
    n_total = msa_binary.shape[0]
    return msa_binary, nl, nc, n_total

def get_embeddings(model, msa_binary):
    """
    Returns the mean of the latent distribution for each sequence in the MSA.
    """
    with torch.no_grad():
        mu, _ = model.encoder(msa_binary)
    return mu

def plot_embeddings(mu, data_path, model_name, MSA_id):
    """
    Plot the embeddings.
    """
    plot_name = os.path.splitext(model_name)[0] + "_embeddings.png"
    plt.figure(figsize=(5, 4))
    plt.scatter(mu[:, 0], mu[:, 1], s=1, alpha=0.9)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('2D Embeddings Visualization')
    plot_dir = get_directory(data_path, MSA_id, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/{plot_name}", bbox_inches='tight')

def save_embeddings(mu, data_path, model_name, MSA_id):
    # Convert to dataframe
    embeddings = pd.DataFrame(mu.numpy(), columns=[f"dim{i}" for i in range(mu.shape[1])])
    # Add names of sequences as a column
    with open(f"{data_path}/seq_names.pkl", 'rb') as file_handle:
        seq_names = pickle.load(file_handle)
    embeddings.insert(0, "id", seq_names)
    # Save to csv
    embeddings_name = os.path.splitext(model_name)[0] + "_embeddings.csv"
    embeddings_dir = get_directory(data_path, MSA_id, "embeddings", data_subfolder=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    embeddings.to_csv(f"{embeddings_dir}/{embeddings_name}", index=False)

def main():
    name_script = sys.argv[0]
    name_json = sys.argv[1]
    print("-" * 50)
    print("Executing " + name_script + " following " + name_json, flush=True)
    print("-" * 50)
    # opening Json file
    json_file = open(name_json)
    data_json = json.load(json_file)
    MSA_id = data_json["MSA_id"]
    data_path = data_json["data_path"]
    model_name = data_json["model_name"]
    plot = data_json["plot"]
    # print the information for training
    print("MSA_id: ", MSA_id)
    print("data_path: ", data_path)
    print("model_name: ", model_name)
    print("plot: ", plot)
    print("-" * 50)

    # get msa
    msa_binary, nl, nc, _ = load_binary_data(data_path)
    # load model
    model_dir = get_directory(data_path, MSA_id, "saved_models")
    model_path = f"{model_dir}/{model_name}"
    ld = int(re.search(r'ld(\d+)', model_name).group(1))
    model = load_model(model_path, nl, nc, nlatent = ld)
    # get embeddings
    mu = get_embeddings(model, msa_binary)
    # plot embeddings
    if plot:
        plot_embeddings(mu, data_path, model_name, MSA_id)
    # save embeddings
    save_embeddings(mu, data_path, model_name, MSA_id)

if __name__ == "__main__":
    main()