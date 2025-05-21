import numpy as np
import torch
import pickle
from autoencoder.model import VAE, EmbedVAE, TVAE
from autoencoder.data import MSA_Dataset

def load_model(model_path, nl, nc=21, 
               num_hidden_units=[500, 100], nlatent=10, 
               one_hot=True, dim_aa_embed=None, trans=False):
    """
    Load the model from the model path.
    """
    if trans:
        model = TVAE(nl=nl, nc=nc, 
                     dim_latent_vars=nlatent,
                     num_hidden_units=num_hidden_units)
    elif one_hot:
        model = VAE(nl=nl, nc = nc, 
                    num_hidden_units=num_hidden_units, 
                    dim_latent_vars=nlatent)
    else:
        model = EmbedVAE(nl=nl, nc=nc, 
                         num_hidden_units=num_hidden_units, 
                         dim_latent_vars=nlatent,
                         dim_aa_embed=dim_aa_embed)
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state_dict)
    return model


def load_data(data_path, weigh_seqs=False, one_hot=True):
    """
    Load the sequence data from the data path as an MSA_Dataset object.
    """
    if one_hot:
        with open(f"{data_path}/seq_msa_binary.pkl", 'rb') as file_handle:
            msa = pickle.load(file_handle)
            nl = msa.shape[1]
            nc = msa.shape[2]
    else:
        with open(f"{data_path}/seq_msa_int.pkl", 'rb') as file_handle:
            msa = pickle.load(file_handle)
            nl = msa.shape[1]
            nc = 21
    with open(f"{data_path}/seq_names.pkl", 'rb') as file_handle:
        seq_names = pickle.load(file_handle)
    if weigh_seqs:
        with open(f"{data_path}/seq_weight.pkl", 'rb') as file_handle:
            seq_weight = pickle.load(file_handle)
    else:
        seq_weight = np.ones(len(seq_names)) / len(seq_names)
    seq_weight = seq_weight.astype(np.float32) 
    assert np.abs(seq_weight.sum() - 1) < 1e-6
    data = MSA_Dataset(msa, seq_weight, seq_names)

    return data, nl, nc