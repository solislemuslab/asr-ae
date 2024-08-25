import torch
from torch.utils.data import Dataset


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
