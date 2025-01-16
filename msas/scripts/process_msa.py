# code adapted from PEVAE paper
import argparse
import pickle
import os
from os import path, makedirs, remove
import numpy as np
import pandas as pd
from Bio import SeqIO


MAX_SEQS = 50_000 # Only used for PF00565, see `get_euk_seqs`
MAX_GAPS_IN_SEQ = 50
MAX_GAPS_IN_POS = 0.2
UNKNOWN = ['-', '.', 'X', 'B', 'Z', 'J'] 
AA = ['R', 'H', 'K',
      'D', 'E',
      'S', 'T', 'N', 'Q',
      'C', 'G', 'P',
      'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
AA_INDEX = {} # mapping from characters to index
for char in UNKNOWN:
    AA_INDEX[char] = 0
for i, char in enumerate(AA):
    AA_INDEX[char] = i + 1


def parse_commands():
    parser = argparse.ArgumentParser(description='This script pre-processes an MSA')
    parser.add_argument('MSA', type=str, help='Path to MSA file')
    parser.add_argument('query_seq_id', type=str, help='ID of a sequence used as reference for filtering')
    parser.add_argument('--real', action='store_true', help='Is MSA from a real PFAM family? Do not use for simulated MSAs')
    args = parser.parse_args()
    return args 
    
def get_euk_seqs(msa_file_path, metadata_file_path):
    """
    This function is currently only used for the MSA of the real family PF00565, which we want to filter to only include eukaryotic sequences.
    The msa file for this family is in stockholm format 
    For all other MSAs, real and simulated, we use the function `get_seqs` to process the sequences.

    REWRITE THE REST OF THIS DOCSTRING AFTER CHANGING THE CODE TO IMPROVE
    The metadata file includes only eukaryotic sequences, so we want to use it to filter the sequences from the MSA that we include in our dictionary.
    Unfortunately, in the part of the file where the actual sequences appear, they are labeled with something that's not the same thing as their accessions from the metadata file. 
    The code in the block "if line[0:4] == "#=GS":" adds to euk_ids only the labels that correspond to accessions that are in the metadata file
    so that later in the for loop, we are able to save only the sequences with these labels
    """
    # Load metadata
    mdata = pd.read_csv(metadata_file_path, delimiter='\t', index_col=False)
    euk_accs = list(mdata.accession) # list of all eukaryotic sequence accessions

    # Initialize dictionaries that will be returned
    euk_ids = {} # sequence label -> sequence accession
    seq_dict = {} # sequence label -> sequence
    count_labels = 0 # number of labels that correspond to eukaryotic accessions
    count_sequences = 0 # number of sequences that have been added to the dictionary
    with open(msa_file_path, 'r') as file_handle:
        for line in file_handle:
            if line[0:4] == "#=GS": 
                if count_labels > MAX_SEQS: # Stop adding sequences to euk_ids once we reach max_seqs
                    continue
                parts = line.split()
                acc, label = parts[-1].split(".")[0], parts[1]
                if acc in euk_accs:
                    count_labels += 1
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
                count_sequences += 1
            # we can break if we've reached the maximum number of sequences
            if count_sequences > MAX_SEQS:
                break
    return seq_dict, euk_ids

def get_seqs(msa_file_path, sim_type):
    """
    Collect the leaf sequences from the MSA file into a dictionary 

    This needs to get fixed to deal with the fact that the format output by SeqGen is currently not being parsed correctly by SeqIO
    """
    seq_dict = {} # sequence id -> sequence
    format_type = {
        'real' : "fasta",
        'coupling_sims' : "phylip-relaxed",
        'independent_sims' : "phylip-sequential" #Weirdly SeqIO is not parsing the MSAs output by SeqGen as (sequential) Phylip format
    }[sim_type]
    with open(msa_file_path, 'r') as file_handle:
        if sim_type != "real": #simulated data 
            if sim_type == "coupling_sims":
                for record in SeqIO.parse(file_handle, format_type):
                    if record.id[0] != "N": #exclude ancestral sequences
                        continue
                    seq_dict[record.id] = str(record.seq).upper()
            else: #this else branch should eventually not be needed once we figure out what's going on with SeqGen format
                seq_data = file_handle.readlines()[1:] # first line is phylip header
                for line in seq_data:
                    if line[0] != "N": #exclude ancestral sequences
                        continue
                    id, seq = line.split()
                    seq_dict[id] = str(seq).upper()
        else: #real data (other than PF00565, see `get_euk_seqs`) is in fasta format and does not have ancestral sequences
            for record in SeqIO.parse(file_handle, format_type):
                seq_dict[record.id] = str(record.seq).upper()
    return seq_dict

def remove_gaps(seq_dict, query_seq_id):
    """
    This does two things at once:
        - Saves sequences in the dictionary as lists of characters instead of as strings
        - Removes from all sequences the positions that are gaps in the query sequences, 
    """
    query_seq = seq_dict[query_seq_id] ## with gaps
    seq_len = len(query_seq)
    is_not_gap = [ s in AA for s in query_seq]
    for seq_id, seq in seq_dict.items():
        seq_dict[seq_id] = [seq[i] for i in range(seq_len) if is_not_gap[i]]

def remove_seqs(seq_dict):
    """
    Remove sequences with too many remaining gaps (called after removing positions that are gaps in the query sequences)
    """
    for k in list(seq_dict.keys()):
        if len([char for char in seq_dict[k] if char in UNKNOWN]) > MAX_GAPS_IN_SEQ:
            seq_dict.pop(k)

def to_numpy(seq_dict, query_seq_id):
    """
    Convert the dictionary of sequences to a numpy array with integer encoding
    Note that now we have to keep track of the labels separately
    We put the query sequence as the top row of the array
    """
    # Inititae list of lists of integers to be converted to np array with first entry the query sequence
    seq_ary = [[AA_INDEX[s] for s in seq_dict[query_seq_id]]] 
    seq_names = [query_seq_id] # Labels
    #iterate through all seq_dict.keys() except the query sequence
    for k in seq_dict.keys() - {query_seq_id}:
        seq_ary.append([AA_INDEX[s] for s in seq_dict[k]])
        seq_names.append(k)    
    seq_ary = np.array(seq_ary)
    return seq_ary, seq_names

def remove_sparse_positions(seq_ary):
    """
    Remove positions where too many sequences have gaps
    """
    pos_idx = []
    for i in range(seq_ary.shape[1]):
        if np.sum(seq_ary[:,i] == 0) <= seq_ary.shape[0]*MAX_GAPS_IN_POS:
            pos_idx.append(i)
    return seq_ary[:, np.array(pos_idx)]

def remove_dupes(seq_ary, seq_names):
    """
    Remove duplicate sequences. 
    Query sequence should always be the first sequence in the array so shouldn't get removed even if it's a duplicate
    """
    seq_ary_new, idx = np.unique(seq_ary, axis = 0, return_index=True)
    seq_names_new = [seq_names[i] for i in idx]
    print(f"Removed following sequences as duplicates: {set(seq_names) - set(seq_names_new)}")
    return seq_ary_new, seq_names_new

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
    K = len(AA) + 1 ## num of classes of aa
    D = np.identity(K)
    num_seq = seq_ary.shape[0]
    len_seq = seq_ary.shape[1]
    seq_ary_binary = np.zeros((num_seq, len_seq, K)) # Binary encoded array representing the processed MSA
    for i in range(num_seq):
        seq_ary_binary[i,:,:] = D[seq_ary[i]]
    return seq_ary_binary

def main():
    #### Preliminary steps and loading in the unprocessed MSA ########
    args = parse_commands()
    msa_file_path = args.MSA
    fam_name, _ = path.splitext(path.basename(msa_file_path))
    if not args.real:
        sim_type = path.dirname(msa_file_path).split("/")[0] #either coupled or independent
        num_seqs = path.dirname(msa_file_path).split("/")[-1] #either 1250 or 5000
        fam_name = fam_name.split("_")[0]
        processed_directory = f"{sim_type}/processed/{num_seqs}/{fam_name}"
    else:
        sim_type = "real"
        processed_directory = f"real/processed/{fam_name}"
    if not path.exists(processed_directory):
        makedirs(processed_directory)
    # Save the mapping between characters and indices
    with open(f"{processed_directory}/aa_index.pkl", 'wb') as file_handle:
        pickle.dump(AA_INDEX, file_handle)

    # Load the sequences from the MSA file
    if fam_name == "PF00565":
        metadata_file_path = f"real/raw/{fam_name}_eukaryotes.tsv"
        seq_dict, euk_ids = get_euk_seqs(msa_file_path, metadata_file_path, args.outgroup_acc)
        with open(f"{processed_directory}/label_accession_mapping.pkl", 'wb') as file_handle:
            pickle.dump(euk_ids, file_handle)
    else: 
        seq_dict = get_seqs(msa_file_path, sim_type)
    
    # ensure that the query accession is in the MSA
    assert args.query_seq_id in seq_dict, f"Query accession {args.query_seq_id} not found in MSA"

    #### Preprocessing #####
    # Step 1: remove all positions that are gaps in the query sequences
    remove_gaps(seq_dict, args.query_seq_id) 

    # Step 2: Remove sequences with too many gaps
    remove_seqs(seq_dict)
    
    # Step 3: Convert to an integer numpy array
    seq_ary_int, seq_names = to_numpy(seq_dict, args.query_seq_id)
    
    # Step 4: remove positions where too many sequences have gaps 
    seq_ary_int = remove_sparse_positions(seq_ary_int)
    # Step 5: remove duplicate sequences
    seq_ary_int, seq_names = remove_dupes(seq_ary_int, seq_names)

    # Step 6: get sequence weights
    seq_weight = weight_seqs(seq_ary_int)

    # Step 7: Convert to a binary (one-hot) encoding
    seq_ary_binary = one_hot_encode(seq_ary_int)

    #### Saving results ####
    # save the sequence labels
    with open(f"{processed_directory}/seq_names.pkl", 'wb') as file_handle:
        pickle.dump(seq_names, file_handle)
    # save the sequence weights
    with open(f"{processed_directory}/seq_weight.pkl", 'wb') as file_handle:
        pickle.dump(seq_weight, file_handle)
    # save integer encoded MSA
    with open(f"{processed_directory}/seq_msa_int.pkl", 'wb') as file_handle:
        pickle.dump(seq_ary_int, file_handle)
    # save one-hot encoded MSA
    with open(f"{processed_directory}/seq_msa_binary.pkl", 'wb') as file_handle:
        pickle.dump(seq_ary_binary, file_handle)
    # Save a character version of our processed MSA (fasta format)
    aa = ["."] + AA # this is so that aa[0] = "." 
    with open(f"{processed_directory}/seq_msa_char.fasta", "w") as f:
        for seq_id, seq in zip(seq_names, seq_ary_int.tolist()):
            decoded_seq = "".join([aa[i] for i in seq])
            f.write(f">{seq_id}\n{decoded_seq}\n")

if __name__ == "__main__":
    main()