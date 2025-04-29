# code adapted from PEVAE paper
import argparse
import pickle
import os
import sys
from os import path, makedirs
import numpy as np
import pandas as pd
from Bio import SeqIO
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilities import constants
from utilities.seq import aa_to_int, invert_dict

## Define global variables for gap-based filtering 
MAX_GAPS_IN_SEQ = constants.MAX_GAPS_IN_SEQ #(relevant for real MSAs)
MAX_GAPS_IN_POS = constants.MAX_GAPS_IN_POS

def parse_commands():
    parser = argparse.ArgumentParser(description='This script pre-processes an MSA')
    parser.add_argument('MSA', type=str, help='Path to MSA file')
    parser.add_argument('--real', action='store_true', help='Is MSA a real PFAM family (as opposed to simulated)?')
    parser.add_argument('--query', type=str, required='--real' in sys.argv, help='ID of a sequence used as reference for filtering')
    args = parser.parse_args()
    return args 
    
def get_euk_seqs(msa_file_path, metadata_file_path):
    """
    This function is currently only used for the MSA of the real family PF00565, for which the MSA file is in Stockholm format 
    and which we filter to include only eukaryotic sequences.
    
    For all other MSAs, real and simulated, we use the function `get_seqs` to obtain the sequences from the MSA.

    How do we filter to include only eukaryotic sequences?
    We can use the metadata file, which includes only eukaryotic sequences.
    The code in the block "if line[0:4] == "#=GS":" adds to euk_ids only the labels that correspond to accessions that are in the metadata file
    so that later in the for loop, we are able to save only the sequences with these labels
    """
    # Load metadata
    mdata = pd.read_csv(metadata_file_path, delimiter='\t', index_col=False)
    euk_accs = list(mdata.accession) # list of all eukaryotic sequence accessions

    # Initialize dictionaries that will be returned
    euk_ids = {} # sequence label -> sequence accession
    seq_dict = {} # sequence label -> sequence
    with open(msa_file_path, 'r') as file_handle:
        for line in file_handle:
            if line[0:4] == "#=GS": # this part of the file contains the labels that correspond to the accessions
                parts = line.split()
                acc, label = parts[-1].split(".")[0], parts[1]
                if acc in euk_accs:
                    euk_ids[label] = acc
                continue
            if line[0] == "#" or line[0] == "/" or line[0] == "":
                continue
            line = line.strip()
            if len(line) == 0:
                continue
            seq_id, seq = line.split()
            # only include sequences if their accession has been found to belong to the list of accessions of eukaryotic species
            if seq_id not in euk_ids: 
                continue
            else:
                seq_dict[seq_id] = seq.upper()
    return seq_dict, euk_ids

def get_seqs(msa_file_path, sim_type):
    """
    Collect the leaf sequences from the MSA file into a dictionary 

    This needs to get fixed to deal with the fact that the format output by SeqGen is currently not being parsed correctly by SeqIO
    """
    seq_dict = {} # sequence id -> sequence
    format = "fasta"
    with open(msa_file_path, 'r') as file_handle:
        if sim_type != "real": #simulated data 
            for record in SeqIO.parse(file_handle, format):
                if not (record.id.startswith("N") and record.id[1:].isdigit()): # exclude ancestral sequences
                    continue
                seq_dict[record.id] = str(record.seq).upper()
        else: #real data (other than PF00565, see `get_euk_seqs`) is in fasta format and does not have ancestral sequences
            for record in SeqIO.parse(file_handle, format):
                seq_dict[record.id] = str(record.seq).upper()
    return seq_dict

def remove_gaps(seq_dict, query_seq_id, aa_symbols = constants.REAL_AA):
    """
    Modifies seq_dict, removing from all sequences the positions that are gaps in the query sequences
    Returns the indices of the preserved positions 
    """
    query_seq = seq_dict[query_seq_id] ## with gaps
    is_not_gap = [ s in aa_symbols for s in query_seq]
    for seq_id, seq in seq_dict.items():
        seq_dict[seq_id] = ''.join([char for char, keep in zip(seq, is_not_gap) if keep])
    return np.where(is_not_gap)[0]

def remove_seqs(seq_dict, max_gaps_in_seq = MAX_GAPS_IN_SEQ):
    """
    Remove sequences with too many remaining gaps (called after removing positions that are gaps in the query sequences)
    """
    nl = len(list(seq_dict.values())[0])
    max_gaps = max_gaps_in_seq * nl
    for k in list(seq_dict.keys()):
        if len([char for char in seq_dict[k] if char in constants.UNKNOWN]) > max_gaps:
            seq_dict.pop(k)

def to_numpy(seq_dict, aa_index):
    """
    Convert the dictionary of sequences to a numpy array with integer encoding based on aa_index
    We put the query sequence as the top row of the array
    Note that now we have to keep track of the labels separately
    """
    seq_ary, seq_names = [], []
    for k in seq_dict.keys():
        seq_ary.append([aa_index[s] for s in seq_dict[k]])
        seq_names.append(k)    
    seq_ary = np.array(seq_ary, dtype=np.uint8)
    return seq_ary, seq_names

def remove_sparse_positions(seq_ary, max_gaps_in_pos = MAX_GAPS_IN_POS):
    """
    Remove positions where too many sequences have gaps
    """
    pos_idx = []
    for i in range(seq_ary.shape[1]):
        if np.sum(seq_ary[:,i] == 0) <= seq_ary.shape[0]*max_gaps_in_pos:
            pos_idx.append(i)
    return seq_ary[:, np.array(pos_idx)], pos_idx

def weight_seqs(seq_ary):
    """
    Calculate weights for each sequence (summing to 1)
    See PEVAE paper Methods section for details
    """
    ## reweighting sequences
    seq_weight = np.zeros(seq_ary.shape)
    for j in range(seq_ary.shape[1]):
        aa_type, aa_counts = np.unique(seq_ary[:,j], return_counts = True)
        num_type = len(aa_type)
        aa_dict = {}
        for a in aa_type:
            aa_dict[a] = aa_counts[list(aa_type).index(a)]
        for i in range(seq_ary.shape[0]):
            seq_weight[i,j] = (1.0/num_type) * (1.0/aa_dict[seq_ary[i,j]])
    tot_weight = np.sum(seq_weight)
    seq_weight = seq_weight.sum(1) / tot_weight 
    return seq_weight

def one_hot_encode(seq_ary):
    """
    Convert the integer encoded array to a binary (one-hot) encoding
    """
    K = len(constants.AA) + 1 ## num of classes of aa
    D = np.identity(K, dtype=np.uint8)
    num_seq = seq_ary.shape[0]
    len_seq = seq_ary.shape[1]
    seq_ary_binary = np.zeros((num_seq, len_seq, K), dtype=np.uint8) # Binary encoded array representing the processed MSA
    for i in range(num_seq):
        seq_ary_binary[i,:,:] = D[seq_ary[i]]
    return seq_ary_binary

def main():
    #### Preliminary steps and loading in the unprocessed MSA ########
    args = parse_commands()
    msa_file_path = args.MSA
    fam_name, _ = path.splitext(path.basename(msa_file_path))
    if not args.real:
        sim_type =  msa_file_path.split(os.sep)[1] #either potts, coupled, or independent
        num_seqs = msa_file_path.split(os.sep)[-2] #either 1250 or 5000
        fam_name = fam_name.split("_")[0]
        processed_directory = f"msas/{sim_type}/processed/{num_seqs}/{fam_name}"
    else:
        sim_type = "real"
        processed_directory = f"msas/real/processed/{fam_name}"
    if not path.exists(processed_directory):
        makedirs(processed_directory)
    
    # Create mapping between from amino-acids to integers (all symbols in constants.UNKNOWN are mapped to 0)
    # TODO: use vocabulary of MSA to create aa_index instead of one size fits all 
    # (can allow for multiple indices for different unknown symbols, for example)
    if args.real:
        aa_index = aa_to_int(constants.REAL_AA, constants.UNKNOWN)
    else:
        aa_index = aa_to_int(constants.AA, constants.UNKNOWN)
    with open(f"{processed_directory}/aa_index.pkl", 'wb') as file_handle:
        pickle.dump(aa_index, file_handle)
    # Load the sequences from the MSA file
    if fam_name == "PF00565":
        metadata_file_path = f"msas/real/raw/PF00565_eukaryotes.tsv"
        seq_dict, euk_ids = get_euk_seqs(msa_file_path, metadata_file_path)
        with open(f"{processed_directory}/label_accession_mapping.pkl", 'wb') as file_handle:
            pickle.dump(euk_ids, file_handle)
    else: 
        seq_dict = get_seqs(msa_file_path, sim_type)
    
    # ensure that the query accession is in the MSA
    if args.real:
        assert args.query in seq_dict, f"Query accession {args.query} not found in MSA"

    #### Preprocessing #####
    # Step 1 and 2, if real: remove all positions that are gaps in the query sequences and the subsequently remove sequences with too many gaps
    if args.real:
        is_not_query_gap = remove_gaps(seq_dict, args.query)
        remove_seqs(seq_dict) 

    # Step 3: Convert to an integer encoding
    seq_ary_int, seq_names = to_numpy(seq_dict, aa_index)
    
    # Step 4: remove positions where too many sequences have gaps 
    seq_ary_int, pos_not_sparse = remove_sparse_positions(seq_ary_int)
    # note that pos_not_sparse indexes positions in the sequences after the gaps from the query sequence have already been removed
    if args.real:
        pos_not_sparse = is_not_query_gap[pos_not_sparse]

    # Step 5: get sequence weights
    seq_weight = weight_seqs(seq_ary_int)

    # Step 6: Convert to a binary (one-hot) encoding
    seq_ary_binary = one_hot_encode(seq_ary_int)

    #### Saving results ####
    # save the sequence labels
    with open(f"{processed_directory}/seq_names.pkl", 'wb') as file_handle:
        pickle.dump(seq_names, file_handle)
    # save the sequence weights
    with open(f"{processed_directory}/seq_weight.pkl", 'wb') as file_handle:
        pickle.dump(seq_weight, file_handle)
    # save the positions in the original sequences that are preserved in the processed MSA
    with open(f"{processed_directory}/pos_preserved.pkl", 'wb') as file_handle:
        pickle.dump(pos_not_sparse, file_handle)
    # save integer encoded MSA
    with open(f"{processed_directory}/seq_msa_int.pkl", 'wb') as file_handle:
        pickle.dump(seq_ary_int, file_handle)
    # save one-hot encoded MSA
    with open(f"{processed_directory}/seq_msa_binary.pkl", 'wb') as file_handle:
        pickle.dump(seq_ary_binary, file_handle)
    # Save a character version of our processed MSA (fasta format) 
    index_aa = invert_dict(aa_index, unknown_symbol = '-') #all unknown characters will be represented by '-'
    with open(f"{processed_directory}/seq_msa_char.fasta", "w") as f:
        for seq_id, seq in zip(seq_names, seq_ary_int.tolist()):
            decoded_seq = "".join([index_aa[i] for i in seq])
            f.write(f">{seq_id}\n{decoded_seq}\n")

if __name__ == "__main__":
    main()