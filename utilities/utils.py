import os
import torch
import numpy as np
import pickle
import sys 
from Bio import SeqIO
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
    """
    Create a dictionary that maps integers to amino acids.
    """
    # In our integer encoding of proteins, we've encoded several different amino acid characters as 0
    # For decoding purposes, we will decode all 0's as '-'
    del aa_index['.'], aa_index['X'], aa_index['B'], aa_index['Z'], aa_index['J']
    idx_to_aa_dict = {}
    for k, v in aa_index.items():
        idx_to_aa_dict[v] = k
    return idx_to_aa_dict   

def filter_fasta(og_path, new_path, keep):
    """
    Writes a new fasta file that only includes the sequences in the keep list.
    If og_path is the same as new_path, the file will be overwritten.
    """
    records_to_keep = []
    with open(og_path, 'r') as og:
        for record in SeqIO.parse(og, "fasta"):
            if record.id in keep:
                records_to_keep.append(record)
    
    with open(new_path, 'w') as new:
        SeqIO.write(records_to_keep, new, "fasta")

def load_data(data_path, weigh_seqs):
    """
    Load the data from the data path.
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