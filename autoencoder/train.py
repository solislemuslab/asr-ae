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
    batch_size = data_json["batch_size"]
    lr = data_json["learning_rate"]
    wd = data_json["weight_decay"]
    latent_dim = data_json["latent_dim"]
    verbose = data_json["verbose"]
    save_model = data_json["save_model"]
    plot_results = data_json["plot_results"]

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
    # The default archictecture for the VAE will be two hidden layers in both the decoder 
    # and encoder, with 256 neurons in each. Can change this by specifying argument
    # num_hidden_units in the constructor
    if use_transformer:
        model = TVAE(nl=nl, nc=nc, dim_latent_vars=latent_dim).to(device)
    else:
        model = VAE(nl=nl, nc=nc, dim_latent_vars=latent_dim).to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # initialize the data loader
    train_idx, valid_idx = train_test_split(range(len(data)), test_size=0.1, random_state=42)
    train_loader = DataLoader(data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    valid_loader = DataLoader(data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(valid_idx))
    with open(f"{model_dir}/valid_idx.pkl", 'wb') as file_handle:
        pickle.dump(valid_idx, file_handle)

    # training the model
    vae_elbo_train, recon_log_train, iwae_elbo_train, recon_acc_val, iwae_elbo_val = [], [], [], [], []
    for epoch in range(num_epochs):
        # Evaluate the IWAE Elbo and reconstruction accuracy on validation data
        ravs, ievs = [], []
        for (msa, _, _) in valid_loader:
            msa = msa.to(device)
            #all_seq_elbos = vae.compute_iwae_elbo(msa, num_samples=1000)
            #ave_elbos = torch.mean(all_seq_elbos).item()
            #ievs.append(ave_elbos)
            all_seq_accs = model.compute_acc(msa)  
            ave_acc = torch.mean(all_seq_accs).item()
            ravs.append(ave_acc)    

        # Train and evaluate IWAE Elbo on training data
        vets, rlts, iets = [], [], []
        for (msa, weight, _) in train_loader:
            # Evaluate the IWAE Eelbo on training data 
            msa, weight = msa.to(device), weight.to(device)
            #all_seq_elbos = vae.compute_iwae_elbo(msa, num_samples=1000)
            #ave_elbos = torch.mean(all_seq_elbos).item()
            #iets.append(ave_elbos)
            # Train on training data with regular VAE loss
            elbo, log_pxgz = model.compute_weighted_elbo(msa, weight)
            vets.append(elbo.item())
            rlts.append(log_pxgz.item())
            loss = (-1) * elbo
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        vae_elbo_train.append(np.mean(vets))
        recon_log_train.append(np.mean(rlts))
        #iwae_elbo_train.append(np.mean(iets))
        recon_acc_val.append(np.mean(ravs))
        #iwae_elbo_val.append(np.mean(ievs))
        print(f"Epoch: {epoch:>4}, "
              f"Training VAE ELBO: {np.mean(vets):>4.2f} "
              f"Training Reconstruction Log Loss: {np.mean(rlts):>4.2f} "
              #f"Training IWAE ELBO: {np.mean(iets):>4.2f} "
              f"Validation Reconstruction Accuracy: {np.mean(ravs):>7.5f} "
              #f"Validation IWAE ELBO: {np.mean(ievs):>4.2f}"
              , flush=True)

    # save the model
    today = date.today()
    model_name = f"model_ld{latent_dim}_wd{wd}_epoch{num_epochs}_{today}.pt"
    if save_model:
        model.cpu()
        torch.save(model.state_dict(), f"{model_dir}/{model_name}")

    # plot learning curve
    if plot_results:
        fig, axs = plt.subplots(1, 2)
        fig.tight_layout()
        
        axs[0].plot(vae_elbo_train, label = "VAE", color = 'r')
        #axs[0].plot(iwae_elbo_train, label = "IWAE", color = 'b')
        axs[0].plot(recon_log_train, label = "Recon log", color = 'g')
        axs[0].set_xlabel('epoch')
        axs[0].legend()
        axs[0].set_title("Training")

        axs[1].plot(recon_acc_val)
        axs[1].set_xlabel('epoch')
        axs[1].set_ylabel("Reconstruction accuracy (Hamming)")
        axs[1].set_title("Validation")

        #save figure
        plot_name = os.path.splitext(model_name)[0] + ".png"
        plot_dir = get_directory(data_path, MSA_id, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/{plot_name}", bbox_inches='tight')
        


if __name__ == '__main__':
    main()
