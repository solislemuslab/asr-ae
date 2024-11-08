from datetime import date
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
import sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from modules.model import VAE, TVAE
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilities.utils import get_directory, load_data 

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
    wd = data_json["weight_decay"]
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
    print("weight_decay: ", wd)
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

    # The default archictecture for the VAE will be two hidden layers in both the decoder 
    # and encoder, with 256 neurons in each. Can change this by specifying argument
    # num_hidden_units in the constructor
    if use_transformer:
        model = TVAE(nl=nl, nc=nc, dim_latent_vars=latent_dim).to(device)
    else:
        model = VAE(nl=nl, nc=nc, dim_latent_vars=latent_dim).to(device)

    optimizer = optim.Adam(model.parameters(), weight_decay=wd)
    # train_idx, test_idx = train_test_split(range(len(data)), test_size=0.1, random_state=42)
    # train_loader = DataLoader(data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    # valid_loader = DataLoader(data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_idx))
    train_data_loader = DataLoader(data, batch_size = batch_size, shuffle = True)

    # training the model
    vae_elbo, recon_log, recon_acc, iwae_elbo = [], [], [], []
    for epoch in range(num_epochs):
        ves, rls, ras, ies = [], [], [], []
        for (msa, weight, _) in train_data_loader:
            msa, weight = msa.to(device), weight.to(device)
            # Evaluate
            model.eval()
            #all_seq_elbos = vae.compute_iwae_elbo(msa, num_samples=100)
            all_seq_accs = model.compute_acc(msa)  
            #ave_elbos = torch.mean(all_seq_elbos).item()
            ave_acc = torch.mean(all_seq_accs).item()
            #ies.append(ave_elbos)
            ras.append(ave_acc)

            # Train
            model.train()
            elbo, log_pxgz = model.compute_weighted_elbo(msa, weight)
            ves.append(elbo.item())
            rls.append(log_pxgz.item())
            loss = (-1) * elbo
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        vae_elbo.append(np.mean(ves))
        recon_log.append(np.mean(rls))
        recon_acc.append(np.mean(ras))
        #iwae_elbo.append(np.mean(ies))
        if verbose: 
            print(("Epoch: {:>4}, "
                   "VAE Elbo: {:>4.2f} "
                   "Recon log: {:>4.2f} "
                   "Recon acc: {:>7.5f} "
                   "IWAE Elbo: NA ").format(epoch, np.mean(ves), np.mean(rls), np.mean(ras)), flush=True)

    # save the model
    today = date.today()
    model_name = f"model_ld{latent_dim}_wd{wd}_epoch{num_epochs}_{today}.pt"
    if save_model:
        model.cpu()
        model_dir = get_directory(data_path, MSA_id, "saved_models")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{model_dir}/{model_name}")

    # plot learning curve
    if plot_results:
        plot_name = os.path.splitext(model_name)[0] + ".png"
        
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(recon_acc, color='r')
        axs[0].set_xlabel('epoch')
        axs[0].set_ylabel(f"Amino acid reconstruction accuracy")

        axs[1].plot(recon_log, label="Log p(x | z)", color='r')
        axs[1].plot(vae_elbo, label="ELBO", color='b')
        axs[1].set_xlabel('epoch')
        axs[1].set_ylabel(f"elbo")
        axs[1].legend()

        # add title
        plt.suptitle(f"Learning curves for VAE trained on {MSA_id}")
        plt.tight_layout()
        #save figure
        plot_dir = get_directory(data_path, MSA_id, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/{plot_name}", bbox_inches='tight')
        


if __name__ == '__main__':
    main()
