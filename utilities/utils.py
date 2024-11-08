import os
import torch
import numpy as np
import pickle
import sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from autoencoder.modules.data import MSA_Dataset

def get_directory(data_path, MSA_id, folder, data_subfolder = False):
    if MSA_id[0:3] == "COG": # this is a simulated dataset
        num_seqs = os.path.dirname(data_path).split("/")[-1]
        dir =  f"{folder}/independent_sims/{num_seqs}/{MSA_id}"
    else:
        dir = f"{folder}/real/{MSA_id}"
    if data_subfolder:
        dir_list = dir.split("/")
        dir_list.insert(1, "data")
        dir = ("/").join(dir_list)
    return dir

def idx_to_aa(aa_index):
    # In our integer encoding of proteins, we've encoded several different amino acid characters as 0
    # For decoding purposes, we will decode all 0's as '-'
    del aa_index['.'], aa_index['X'], aa_index['B'], aa_index['Z'], aa_index['J']
    idx_to_aa_dict = {}
    for k, v in aa_index.items():
        idx_to_aa_dict[v] = k
    return idx_to_aa_dict

def to_fasta(f_in, f_out, ids_included = True, keep = False):
    with open(f_in) as in_file, open(f_out, "w") as out_file:
        for idx, line in enumerate(in_file):
            if ids_included:
                id, seq = line.split()
                if keep and id not in keep:
                    continue
                out_file.write(f">{id}\n{seq}\n")
            else:
                out_file.write(f">Seq{idx}\n{line.strip()}\n")                

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