import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

class MSA_Dataset(Dataset):
    """
    Dataset class for multiple sequence alignment.
    """

    def __init__(self, seq_msa_binary, seq_weight, seq_keys):
        """
        seq_msa_binary: a three-dimensional tensor.
                        size: [num_of_sequences, length_of_msa, num_amino_acid_types]
        seq_weight: one dimensional tensor.
                    size: [num_sequences].
                    Weights for sequences in a MSA.
                    The sum of seq_weight has to be equal to 1 when training latent space models using VAE
        seq_keys: name of sequences in MSA
        """
        super(MSA_Dataset).__init__()
        self.seq_msa_binary = seq_msa_binary.to(torch.float32)  # for training
        self.seq_weight = seq_weight
        self.seq_keys = seq_keys

    def __len__(self):
        assert (self.seq_msa_binary.shape[0] == len(self.seq_weight))
        assert (self.seq_msa_binary.shape[0] == len(self.seq_keys))
        return self.seq_msa_binary.shape[0]

    def __getitem__(self, idx):
        return self.seq_msa_binary[idx, :, :], self.seq_weight[idx], self.seq_keys[idx]

def load_data(data_path, weigh_seqs):
    """
    Load the sequence data from the data path as an MSA_Dataset object.
    """
    with open(f"{data_path}/seq_msa_binary.pkl", 'rb') as file_handle:
        msa_binary = torch.tensor(pickle.load(file_handle))
    nl = msa_binary.shape[1]
    nc = msa_binary.shape[2]

    with open(f"{data_path}/seq_names.pkl", 'rb') as file_handle:
        seq_names = pickle.load(file_handle)
    if weigh_seqs:
        with open(f"{data_path}/seq_weight.pkl", 'rb') as file_handle:
            seq_weight = pickle.load(file_handle)
    else:
        seq_weight = np.ones(len(seq_names)) / len(seq_names)
    seq_weight = seq_weight.astype(np.float32) 
    assert np.abs(seq_weight.sum() - 1) < 1e-6
    data = MSA_Dataset(msa_binary, seq_weight, seq_names)

    return data, nl, nc