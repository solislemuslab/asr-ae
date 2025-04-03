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
from autoencoder.model import VAE, TVAE, EmbedVAE
from utilities.vae import load_data
from utilities.utils import get_directory

def train(model, device, train_loader, optimizer, epoch, verbose):
    """
    Define how to do an epoch of training    
    """
    model.train()
    running_elbo = []

    for batch_idx, (msa, weight, _) in enumerate(train_loader):
        msa, weight = msa.to(device), weight.to(device)
        optimizer.zero_grad()
        elbo, _ = model.compute_weighted_elbo(msa, weight)
        loss = (-1)*elbo
        loss.backward()
        optimizer.step()
        elbo_scalar = elbo.item()
        if verbose:
            print("Epoch: {:>4}, Step: {:>4}, loss: {:>4.2f}".format(epoch, batch_idx, elbo_scalar),
                  flush=True)
        running_elbo.append(elbo_scalar)

    return running_elbo

def eval(model, device, valid_loader, n_samples = 100):
    """
    Define how to do an epoch of evaluation
    """
    model.eval()
    elbos = []
    log_pxgzs = []
    elbo_iwaes = []
    accs = []
    with torch.no_grad():
        for (msa, weight, _) in valid_loader:
            msa = msa.to(device)
            # compute elbo loss with normal elbo
            elbo, log_pxgz = model.compute_weighted_elbo(msa, weight)
            elbos.append(elbo.item())
            log_pxgzs.append(log_pxgz.item())

            # compute elbo loss with iwae elbo
            all_iwae_elbos = model.compute_iwae_elbo(msa, n_samples)
            ave_iwae_elbo = torch.mean(all_iwae_elbos).item()
            elbo_iwaes.append(ave_iwae_elbo)

            # compute reconstruction accuracy
            all_accs = model.compute_acc(msa)
            ave_acc = torch.mean(all_accs).item()
            accs.append(ave_acc)


    return elbos, log_pxgzs, elbo_iwaes, accs

def main():

    # opening Json file
    name_json = sys.argv[1]
    with open(name_json, 'r') as json_file:
        data_json = json.load(json_file)
    # loading the input data from the json file
    data_path = data_json["data_path"]
    training_config = data_json["training"]
    one_hot = training_config["one_hot"]
    weigh_seqs = training_config["weigh_seqs"]
    use_transformer = training_config["use_transformer"]
    dim_aa_embed = training_config["dim_aa_embed"] # only relevant if one_hot is False
    num_epochs = training_config["num_epochs"]
    num_hidden_units = training_config["num_hidden_units"]
    batch_size = training_config["batch_size"]
    lr = training_config["learning_rate"]
    wd = training_config["weight_decay"]
    latent_dim = training_config["latent_dim"]
    validate = training_config["validate"]
    iwae_num_samples = training_config["iwae_num_samples"]
    verbose = training_config["verbose"]
    save_model = training_config["save_model"]
    plot_results = training_config["plot_results"]  
  
    # create model path
    today = date.today()
    layers_str = "-".join([str(l) for l in num_hidden_units])
    aa_embed_str = f"aaembed{dim_aa_embed}_" if not one_hot else ""
    model_configs=f"model_{aa_embed_str}layers{layers_str}_ld{latent_dim}_wd{wd}_epoch{num_epochs}"
    model_name = f"{model_configs}_{today}.pt"
    model_dir = get_directory(data_path, "saved_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    
    # check if model with same configs (ignoring date) already exists and only proceed if it doesn't
    existing_models = [f for f in os.listdir(model_dir) if f.startswith(model_configs)]
    if existing_models:
        print(f"Model has already been trained and is saved at {model_dir} with name\n{existing_models[0]}")
        return
    else:
        print("Beginning training...", flush=True)
    
    # load the dataset
    data, nl, nc = load_data(data_path, weigh_seqs=weigh_seqs, one_hot=one_hot)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    print("-" * 50)

    # initialize the model
    if use_transformer:
        model = TVAE(nl=nl, nc=nc, 
                     dim_latent_vars=latent_dim,
                     num_hidden_units=num_hidden_units).to(device)
    elif one_hot:
        model = VAE(nl=nl, nc=nc, 
                    dim_latent_vars=latent_dim,
                    num_hidden_units=num_hidden_units).to(device)
    else:
        model = EmbedVAE(nl=nl, nc=nc, dim_aa_embed=dim_aa_embed,
                        dim_latent_vars=latent_dim,
                        num_hidden_units=num_hidden_units).to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # initialize the data loader
    train_idx, valid_idx = train_test_split(range(len(data)), test_size=0.1, random_state=42)
    train_loader = DataLoader(data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    valid_loader = DataLoader(data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(valid_idx))
    with open(f"{model_dir}/valid_idx.pkl", 'wb') as file_handle:
        pickle.dump(valid_idx, file_handle)

    # training the model
    # TODO: Implement early stopping based on either val_accs or val_log_pxgzs
    train_elbos, val_elbos, val_log_pxgzs, val_iwae_elbos, val_accs = [], [], [], [], []
    for epoch in range(num_epochs):
        # Validation metrics
        print(f"Epoch {epoch}:", end=' ', flush=True)
        if validate:
            val_elbos_on_batches, val_log_pxgzs_on_batches, val_elbo_iwae_on_batches, val_accs_on_batches = eval(
                model, device, valid_loader, iwae_num_samples)
            epoch_val_elbo = np.mean(val_elbos_on_batches)
            epoch_val_log_pxgz = np.mean(val_log_pxgzs_on_batches)
            epoch_val_elbo_iwae = np.mean(val_elbo_iwae_on_batches)
            epoch_val_acc = np.mean(val_accs_on_batches)
            val_elbos.append(epoch_val_elbo)
            val_log_pxgzs.append(epoch_val_log_pxgz)
            val_iwae_elbos.append(epoch_val_elbo_iwae)
            val_accs.append(epoch_val_acc)
            print(f"Validation elbo: {epoch_val_elbo}", end=', ', flush=True)
            print(f"Validation accuracy: {epoch_val_acc}", end=', ', flush=True)
        # Training metrics
        batch_elbos = train(model, device, train_loader, optimizer, epoch, verbose)
        epoch_train_elbo = np.mean(batch_elbos)
        train_elbos.append(epoch_train_elbo)
        print(f"Training elbo {epoch}: {epoch_train_elbo}", flush=True)

    # save the model
    if save_model:
        model.cpu()
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_dir} with name\n{model_name}")

    # plot learning curve
    if plot_results:
        plt.plot(train_elbos[1:], label="Training elbos", color='y')
        if validate:
            plt.plot(val_elbos[1:], label="Validation elbos", color='r')
            plt.plot(val_log_pxgzs[1:], label="Validation log reconstruction probs", color='g')
            plt.plot(val_iwae_elbos[1:], label="Validation IWAE elbos", color='b')
        plt.xlabel("Epoch")
        plt.legend()

        #save figure
        plot_name = os.path.splitext(model_name)[0] + ".png"
        plot_dir = get_directory(data_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/{plot_name}", bbox_inches='tight')

if __name__ == '__main__':
    main()
