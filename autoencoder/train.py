from datetime import date
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
import sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from modules_gd.model import VAE, TVAE
from modules_gd.data import MSA_Dataset


def load_data(data_path):
    """
    Load the data from the data path.
    """
    with open(f"{data_path}/seq_msa_binary.pkl", 'rb') as file_handle:
        msa_binary = torch.tensor(pickle.load(file_handle))
    nl = msa_binary.shape[1]
    nc = msa_binary.shape[2]

    with open(f"{data_path}/seq_names.pkl", 'rb') as file_handle:
        seq_names = pickle.load(file_handle)

    with open(f"{data_path}/seq_weight.pkl", 'rb') as file_handle:
        seq_weight = pickle.load(file_handle)
    seq_weight = seq_weight.astype(np.float32)
    assert np.abs(seq_weight.sum() - 1) < 1e-6

    data = MSA_Dataset(msa_binary, seq_weight, seq_names)

    return data, nl, nc


# Define how to do an epoch of training
def train(model, device, train_loader, optimizer, epoch, verbose):
    model.train()
    running_elbo = []

    for batch_idx, (msa, weight, _) in enumerate(train_loader):
        msa, weight = msa.to(device), weight.to(device)
        optimizer.zero_grad()
        loss = (-1) * model.compute_weighted_elbo(msa, weight)
        loss.backward()
        optimizer.step()
        elbo_scalar = -loss.item()
        if verbose:
            print("Epoch: {:>4}, Step: {:>4}, loss: {:>4.2f}".format(epoch, batch_idx, elbo_scalar), flush=True)
        running_elbo.append(elbo_scalar)

    return running_elbo


# Define how to evaluate the model on the validation data
def eval(model, device, valid_loader, recon=False):
    model.eval()
    elbos = []
    if recon:
        recon_accs = []
    with torch.no_grad():
        for (msa, _, _) in valid_loader:
            msa = msa.to(device)
            # compute elbo loss
            elbo = model.compute_elbo_with_multiple_samples(msa,
                                                            100)  # how many samples to use for IWAE estimate of ELBO
            elbo_scalar = torch.mean(elbo).item()
            elbos.append(elbo_scalar)

            if recon:
                # compute proportion of amino acids correctly reconstructed
                real = torch.argmax(msa, -1)
                mu, _ = model.encoder(msa)
                p = torch.exp(model.decoder(mu))
                preds = torch.argmax(p, -1)
                recon_acc = torch.sum(real == preds) / real.nelement()
                recon_acc_scalar = recon_acc.data.item()
                recon_accs.append(recon_acc_scalar)

    return elbos, recon_accs


def main():
    name_script = sys.argv[0]
    name_json = sys.argv[1]
    print("-" * 50)
    print("Executing " + name_script + " following " + name_json, flush=True)
    print("-" * 50)

    # opening Json file
    json_file = open(name_json)
    data_json = json.load(json_file)
    # loading the input data from the json file
    MSA_id = data_json["MSA_id"]
    data_path = data_json["data_path"]
    use_transformer = data_json["use_transformer"]
    num_epochs = data_json["num_epochs"]
    batch_size = data_json["batch_size"]
    # learning_rate = data_json["learning_rate"]
    latent_dim = data_json["latent_dim"]
    verbose = data_json["verbose"]
    save_model = data_json["save_model"]
    plot_results = data_json["plot_results"]

    # print the information for training
    print("MSA_id: ", MSA_id)
    print("data_path: ", data_path)
    print("use_transformer: ", use_transformer)
    print("num_epochs: ", num_epochs)
    print("batch_size: ", batch_size)
    # print("learning_rate: ", learning_rate)
    print("latent_dim: ", latent_dim)
    print("verbose: ", verbose)
    print("save_model: ", save_model)
    print("plot_results: ", plot_results)
    print("-" * 50)

    # load the data and initialize the data loader
    data, nl, nc = load_data(data_path)

    # initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    print("-" * 50)

    if use_transformer:
        model = TVAE(nl=nl, nc=nc, dim_latent_vars=latent_dim).to(device)
    else:
        model = VAE(nl=nl, nc=nc, dim_latent_vars=latent_dim).to(device)

    optimizer = optim.Adam(model.parameters())
    train_idx, test_idx = train_test_split(range(len(data)), test_size=0.1, random_state=42)
    train_loader = DataLoader(data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    test_loader = DataLoader(data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_idx))

    # training the model
    train_elbos = []
    test_elbos = []
    test_recon_accs = []
    for epoch in range(num_epochs):
        batch_elbos, batch_recon_accs = eval(model, device, test_loader, recon=True)
        epoch_test_elbo, epoch_test_recon_acc = np.mean(batch_elbos), np.mean(batch_recon_accs)
        test_elbos.append(epoch_test_elbo)
        test_recon_accs.append(epoch_test_recon_acc)
        print(f"Test elbo for epoch {epoch}: {epoch_test_elbo}")
        print(f"Test reconstruction accuracy for fold epoch {epoch}: {epoch_test_recon_acc}")
        batch_elbos = train(model, device, train_loader, optimizer, epoch, verbose)
        epoch_train_elbo = np.mean(batch_elbos)
        train_elbos.append(epoch_train_elbo)
        print(f"Training elbo for epoch {epoch}: {epoch_train_elbo}")

    # save the model
    if save_model:
        os.makedirs(f"saved_models/{MSA_id}", exist_ok=True)
        model.cpu()
        today = date.today()
        torch.save(model.state_dict(), f"saved_models/{MSA_id}/model_{today}.pt")

    # plot learning curve
    if plot_results:
        fig, axs = plt.subplots(1, 2)

        axs[0].plot(test_recon_accs[1:], color='r')
        axs[0].set_xlabel('epoch')
        axs[0].set_ylabel(f"Amino acid reconstruction accuracy")

        axs[1].plot(test_elbos[1:], label="test", color='r')
        axs[1].plot(train_elbos[1:], label="train", color='b')
        axs[1].set_xlabel('epoch')
        axs[1].set_ylabel(f"elbo")
        axs[1].legend()

        # add title
        plt.suptitle(f"Learning curve for VAE trained on {MSA_id}")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
