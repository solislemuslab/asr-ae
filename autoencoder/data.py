import torch
from torch.utils.data import Dataset

# TODO: Change this so that it only stores MSAs with integer index encodings of amino acids,
# implementing a transform in the __getitem__ method to do one-hot encoding on the fly
class MSA_Dataset(Dataset):
    """
    Dataset class for multiple sequence alignment.
    """
    def __init__(self, msa, seq_weight, seq_keys):
        """
        msa: either a three-dimensional numpy array (num_sequences, length_of_sequences, num_amino_acid_types) 
        with one-hot encodings of amino acids or a two-dinensional numpy array (num_sequences, length_of_sequences)
        of integer index encodings of amino acids 
        
        seq_weight: array of weights of sequences in MSA. The sum should equal 1.
        
        seq_keys: array of names of sequences in MSA
        """
        super(MSA_Dataset).__init__()
        assert (len(msa.shape) == 2 or len(msa.shape) == 3)
        if len(msa.shape) == 3:
            assert (msa.shape[2] in [20, 21])  # 20 for natural amino acids, 21 for natural amino acids + gap
        assert (msa.shape[0] == len(seq_weight))
        assert (msa.shape[0] == len(seq_keys))
        if len(msa.shape) == 3:
            self.msa= torch.tensor(msa).to(torch.float32)  # We matrix multiply one-hot encodings so we need them as floats
        else:
            self.msa = torch.tensor(msa).to(torch.int64)  # We index with these so we need them as ints
        self.seq_weight = seq_weight
        self.seq_keys = seq_keys

    def __len__(self):
        return self.msa.shape[0]

    def __getitem__(self, idx):
        return self.msa[idx], self.seq_weight[idx], self.seq_keys[idx]

