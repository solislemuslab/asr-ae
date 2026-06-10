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


def compute_metrics(model, loader, n_iwae_samples):
    """Compute IWAE ELBO, log p(x|z), and Hamming accuracy on a data loader."""
    all_iwae_elbos = []
    all_log_pxgzs = []
    all_accs = []
    with torch.no_grad():
        for (msa, weight, _) in tqdm(loader):
            msa, weight = msa.to(DEVICE), weight.to(DEVICE)
            # IWAE ELBO
            batch_elbos = model.compute_iwae_elbo(msa, n_iwae_samples)
            all_iwae_elbos.append(batch_elbos.cpu().numpy())
            # log p(x|z) and ELBO
            _, log_pxgz = model.compute_weighted_elbo(msa, weight)
            all_log_pxgzs.append(log_pxgz.cpu().item())
            # Hamming accuracy
            batch_accs = model.compute_acc(msa)
            all_accs.append(batch_accs.cpu().numpy())
    iwae_elbos = np.concatenate(all_iwae_elbos, axis=0)
    accs = np.concatenate(all_accs, axis=0)
    return {
        "iwae_elbo": np.mean(iwae_elbos),
        "log_pxgz": np.mean(all_log_pxgzs),
        "acc": np.mean(accs),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute IWAE ELBO, log p(x|z), and Hamming accuracy. "
                    "Reports train/validation split for models trained with held-out data, "
                    "or full-dataset metrics for models trained with all data.")

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
    use_all_data = "-alldata" in args.model_name

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

    with open(f"{args.data_path}/seq_msa_int.pkl", 'rb') as f:
        all_seqs_int = pickle.load(f)

    if use_all_data:
        full_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)

        print(f"Model: {args.model_name}")
        print(f"Dataset size: {len(data)}")
        print("=" * 60)

        print("Computing metrics on full dataset...")
        metrics = compute_metrics(model, full_loader, args.n_samples)

        consensus = np.array([np.bincount(all_seqs_int[:, i]).argmax() for i in range(all_seqs_int.shape[1])])
        consensus_acc = np.mean(all_seqs_int == consensus[None, :])

        print("=" * 60)
        print(f"{'Metric':<25} {'Full dataset':>12}")
        print("-" * 60)
        print(f"{'IWAE ELBO':<25} {metrics['iwae_elbo']:>12.2f}")
        print(f"{'Log p(x|z)':<25} {metrics['log_pxgz']:>12.2f}")
        print(f"{'Hamming accuracy':<25} {metrics['acc']:>12.4f}")
        print(f"{'Consensus accuracy':<25} {consensus_acc:>12.4f}")
    else:
        # get indices of train and validation sets
        with open(f"{model_dir}/valid_idx.pkl", 'rb') as file_handle:
            valid_idx = pickle.load(file_handle)
        all_idx = set(range(len(data)))
        train_idx = sorted(all_idx - set(valid_idx))

        # Set up data loaders
        train_loader = DataLoader(data, batch_size=BATCH_SIZE,
                                  sampler=torch.utils.data.SubsetRandomSampler(train_idx))
        valid_loader = DataLoader(data, batch_size=BATCH_SIZE,
                                  sampler=torch.utils.data.SubsetRandomSampler(valid_idx))

        print(f"Model: {args.model_name}")
        print(f"Train set size: {len(train_idx)}, Validation set size: {len(valid_idx)}")
        print("=" * 60)

        print("Computing metrics on training set...")
        train_metrics = compute_metrics(model, train_loader, args.n_samples)
        print("Computing metrics on validation set...")
        valid_metrics = compute_metrics(model, valid_loader, args.n_samples)

        # Consensus from training sequences only
        train_seqs = all_seqs_int[train_idx]
        consensus = np.array([np.bincount(train_seqs[:, i]).argmax() for i in range(train_seqs.shape[1])])
        train_consensus_acc = np.mean(train_seqs == consensus[None, :])
        val_seqs = all_seqs_int[list(valid_idx)]
        val_consensus_acc = np.mean(val_seqs == consensus[None, :])

        print("=" * 60)
        print(f"{'Metric':<25} {'Train':>12} {'Validation':>12}")
        print("-" * 60)
        print(f"{'IWAE ELBO':<25} {train_metrics['iwae_elbo']:>12.2f} {valid_metrics['iwae_elbo']:>12.2f}")
        print(f"{'Log p(x|z)':<25} {train_metrics['log_pxgz']:>12.2f} {valid_metrics['log_pxgz']:>12.2f}")
        print(f"{'Hamming accuracy':<25} {train_metrics['acc']:>12.4f} {valid_metrics['acc']:>12.4f}")
        print(f"{'Consensus accuracy':<25} {train_consensus_acc:>12.4f} {val_consensus_acc:>12.4f}")
