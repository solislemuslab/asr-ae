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
from modules.model import VAE, TVAE
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilities.utils import get_directory, load_data 

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

    with torch.no_grad():
        for (msa, weight, _) in valid_loader:
            msa = msa.to(device)
            # compute elbo loss with normal elbo
            elbo, log_pxgz = model.compute_weighted_elbo(msa, weight)
            elbos.append(elbo.item())
            log_pxgzs.append(log_pxgz.item())

            # compute elbo loss with iwae elbo
            all_iwae_elbos = model.compute_iwae_elbo(msa, n_samples)
            ave_iwae_elbos = torch.mean(all_iwae_elbos).item()
            elbo_iwaes.append(ave_iwae_elbos)

    return elbos, log_pxgzs, elbo_iwaes

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
    weigh_seqs = data_json["weigh_seqs"]
    use_transformer = data_json["use_transformer"]
    num_epochs = data_json["num_epochs"]
    num_hidden_units = data_json["num_hidden_units"]
    batch_size = data_json["batch_size"]
    lr = data_json["learning_rate"]
    wd = data_json["weight_decay"]
    latent_dim = data_json["latent_dim"]
    validate = data_json["validate"]
    iwae_num_samples = data_json["iwae_num_samples"]
    verbose = data_json["verbose"]
    save_model = data_json["save_model"]
    plot_results = data_json["plot_results"]
    # close the json file   
    json_file.close()
    
    # print the information for training
    print("MSA_id: ", MSA_id)
    print("data_path: ", data_path)
    print("weigh_seqs?", weigh_seqs)
    print("use_transformer: ", use_transformer)
    print("num_epochs: ", num_epochs)
    print("batch_size: ", batch_size)
    # print("learning_rate: ", learning_rate)
    print("weight_decay: ", wd)
    print("num_hidden_units: ", num_hidden_units)
    print("latent_dim: ", latent_dim)
    print("validate: ", validate)
    print("iwae_num_samples: ", iwae_num_samples)
    print("verbose: ", verbose)
    print("save_model: ", save_model)
    print("plot_results: ", plot_results)
    print("-" * 50)

    # create the directory to save the model
    model_dir = get_directory(data_path, MSA_id, "saved_models")
    os.makedirs(model_dir, exist_ok=True)

    # load the dataset
    data, nl, nc = load_data(data_path, weigh_seqs)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    print("-" * 50)

    # initialize the model
    if use_transformer:
        model = TVAE(nl=nl, nc=nc, 
                     num_hidden_units=num_hidden_units, 
                     dim_latent_vars=latent_dim).to(device)
    else:
        model = VAE(nl=nl, nc=nc, 
                    num_hidden_units=num_hidden_units, 
                    dim_latent_vars=latent_dim).to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # initialize the data loader
    train_idx, valid_idx = train_test_split(range(len(data)), test_size=0.1, random_state=42)
    train_loader = DataLoader(data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    valid_loader = DataLoader(data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(valid_idx))
    with open(f"{model_dir}/valid_idx.pkl", 'wb') as file_handle:
        pickle.dump(valid_idx, file_handle)

    # training the model
    train_elbos, val_elbos, val_log_pxgzs, val_iwae_elbos = [], [], [], []
    for epoch in range(num_epochs):
        # Validation metrics
        if validate:
            val_elbos_on_batches, val_log_pxgzs_on_batches, val_elbo_iwae_on_batches = eval(
                model, device, valid_loader, iwae_num_samples)
            epoch_val_elbo = np.mean(val_elbos_on_batches)
            epoch_val_log_pxgz = np.mean(val_log_pxgzs_on_batches)
            epoch_val_elbo_iwae = np.mean(val_elbo_iwae_on_batches)
            val_elbos.append(epoch_val_elbo)
            val_log_pxgzs.append(epoch_val_log_pxgz)
            val_iwae_elbos.append(epoch_val_elbo_iwae)
            print(f"Validation elbo for epoch {epoch}: {epoch_val_elbo}", flush=True)
        # Training metrics
        batch_elbos = train(model, device, train_loader, optimizer, epoch, verbose)
        epoch_train_elbo = np.mean(batch_elbos)
        train_elbos.append(epoch_train_elbo)
        print(f"Training elbo for epoch {epoch}: {epoch_train_elbo}", flush=True)

    # save the model
    today = date.today()
    layers = "-".join([str(l) for l in num_hidden_units])
    model_name = f"model_layers{layers}_ld{latent_dim}_wd{wd}_epoch{num_epochs}_{today}.pt"
    if save_model:
        model.cpu()
        torch.save(model.state_dict(), f"{model_dir}/{model_name}")

    # plot learning curve
    if plot_results:
        plt.plot(train_elbos[:-1], label="Training elbos", color='y')
        if validate:
            plt.plot(val_elbos[1:], label="Validation elbos", color='r')
            plt.plot(val_log_pxgzs[1:], label="Validation log reconstruction probs", color='g')
            plt.plot(val_iwae_elbos[1:], label="Validation IWAE elbos", color='b')
        plt.xlabel("Epoch")
        plt.legend()

        #save figure
        plot_name = os.path.splitext(model_name)[0] + ".png"
        plot_dir = get_directory(data_path, MSA_id, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/{plot_name}", bbox_inches='tight')

if __name__ == '__main__':
    main()
