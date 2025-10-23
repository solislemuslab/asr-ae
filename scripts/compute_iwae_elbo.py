import argparse
import os
from tqdm import tqdm
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilities.vae import load_model, load_data
from utilities.utils import get_directory, parse_model_name

BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute IWAE Elbo of model on validation set")

    parser.add_argument("data_path", type=str, help="Path to data directory")
    parser.add_argument("model_name", type=str,
                        help="Name of the trained model file")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of importance samples for IWAE ELBO")
    args = parser.parse_args()

    # Get model hyperparameters from name
    (is_trans,
        ld,
        num_hidden_units,
        dim_aa_embed,
        one_hot
     ) = parse_model_name(args.model_name)
    ding_model = args.model_name.startswith("ding")

    # load data
    data, nl, nc = load_data(args.data_path, one_hot=one_hot)

    # load model
    model_dir = get_directory(args.data_path, "saved_models")
    model_path = os.path.join(model_dir, args.model_name)
    model = load_model(model_path, nl=nl, nc=nc, ding_model=ding_model,
                       num_hidden_units=num_hidden_units, nlatent=ld,
                       one_hot=one_hot, dim_aa_embed=dim_aa_embed, trans=is_trans)
    model = model.to(DEVICE)
    model.eval()

    # get indices of validation set
    with open(f"{model_dir}/valid_idx.pkl", 'rb') as file_handle:
        valid_idx = pickle.load(file_handle)
    
    # Set up validation data loader
    valid_loader = DataLoader(data, batch_size=BATCH_SIZE,
                              sampler=torch.utils.data.SubsetRandomSampler(valid_idx))
    
    # Compute IWAE ELBO on validation set
    all_elbos = []
    with torch.no_grad():
        for (msa, _, _) in tqdm(valid_loader):
            msa = msa.to(DEVICE)
            batch_elbos = model.compute_iwae_elbo(msa, args.n_samples)
            all_elbos.append(batch_elbos.cpu().numpy())

    all_elbos = np.concatenate(all_elbos, axis=0)
    mean_iwae_elbo = np.mean(all_elbos)
    print(f"IWAE ELBO on validation set with {args.n_samples} samples: {mean_iwae_elbo}")
