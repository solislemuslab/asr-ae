import os
import re
import numpy as np
from typing import List, Tuple, Dict, Optional
from numpy.typing import NDArray
from Bio import SeqIO
from utilities import constants


def get_directory(data_path, folder, data_subfolder = False):
    """
    Take data path and MSA id and return the relevant directory
    """
    MSA_id = os.path.basename(data_path)
    if MSA_id[0:3] == "COG" or MSA_id == "pevae": # this is a simulated dataset
        sim_type = os.path.dirname(data_path).split("/")[1] 
        num_seqs = os.path.dirname(data_path).split("/")[-1] 
        dir =  f"{folder}/{sim_type}/{num_seqs}/{MSA_id}"
    else:
        dir = f"{folder}/real/{MSA_id}"
    if data_subfolder:
        dir_list = dir.split("/")
        dir_list.insert(1, "data")
        dir = ("/").join(dir_list)
    return dir

def parse_model_name(model_name):
    """
    Parse the model name to extract hyperparameters.
    """
    # Latent dimension of VAE
    # use re to check if model name starts with "trans"
    if re.match(r'^trans', model_name):
        is_transformer = True
    else:
        is_transformer = False
    ld = int(re.search(r'ld(\d+)', model_name).group(1))
    # Number of hidden units in VAE
    layers_match = re.search(r'layers(\d+(\-\d+)*)', model_name)
    num_hidden_units = [int(size) for size in layers_match.group(1).split('-')]
    # aa embedding dimension will be present in the model name if model is an EmbedVAE
    aa_embed_match = re.search(r'aaembed(\d+)', model_name)
    dim_aa_embed = int(aa_embed_match.group(1)) if aa_embed_match else None
    one_hot = not aa_embed_match

    return is_transformer, ld, num_hidden_units, dim_aa_embed, one_hot

def get_real_internals(
    msa_path: str,
    aa_index: Dict[str, int],
    sorted_ids: Optional[List[str]] = None,
    pos_preserved: Optional[List[int]] = None,
    exclude_root: bool = True
) -> Tuple[NDArray[np.int_], List[str]]:
    """
    Retrieves the internal sequences from the MSA file
    Returns tuple of integer encoded MSA (numpy array) and list of ids 
    """
    format = "fasta"
    real_seqs_dict = {}
    with open(msa_path, 'r') as msa:
        for record in SeqIO.parse(msa, format):
            # exclude leaf sequences (begin with N)
            if record.id[0] == "N" or (exclude_root and record.id == "__root__"): 
                continue  
            seq = str(record.seq)
            if pos_preserved is not None: # keep only positions that were preserved in the processed MSA
                seq = "".join([seq[pos] for pos in pos_preserved])
            real_seqs_dict[record.id] = seq
    # order true ancestral sequences according to anc_id
    if sorted_ids is not None:
        real_seqs = [real_seqs_dict[id] for id in sorted_ids]
        ids = sorted_ids
    else:
        real_seqs = list(real_seqs_dict.values())
        ids = list(real_seqs_dict.keys())
    # convert to integers for comparison with reconstructed sequences
    real_seqs_int = np.array([[aa_index[aa] for aa in seq] for seq in real_seqs])
    return real_seqs_int, ids

def one_hot_encode(seq_ary: NDArray[np.int_]) -> NDArray[np.uint8]: 
    """
    Convert the integer encoded array to a binary (one-hot) encoding
    """
    # Gaps are represented by 0. If there are no gaps, we can use 20 classes for amino acids.
    gapped = np.any(seq_ary == 0) 
    nc = 21 if gapped else 20
    D = np.identity(nc, dtype=np.uint8)
    num_seq = seq_ary.shape[0]
    len_seq = seq_ary.shape[1]
    seq_ary_binary = np.zeros((num_seq, len_seq, nc), dtype=np.uint8) 
    for i in range(num_seq):
        idx_aas = seq_ary[i]
        if not gapped: # convert from 1-20 to 0-19
            idx_aas = idx_aas - 1
        seq_ary_binary[i,:,:] = D[idx_aas]
    return seq_ary_binary