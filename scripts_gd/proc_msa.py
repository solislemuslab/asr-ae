# code adapted from PEVAE paper
import argparse
import pickle
from os import path, makedirs, remove
import numpy as np
import pandas as pd
from Bio import SeqIO
from sys import exit

MAX_SEQS = 50_000
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
    """
    Parse command line arguments 
    Args:
        - Path to MSA file (required)
        - Reference (aka query) sequence id (required)
        - Whether the MSA is simulated (optional, default is False)
        - Whether to filter out non-eukaryotic sequences (optional, default is False)
    Additional optional args if we're filtering out non-eukarytoic sequences:
        - Metadata file path (default is inferred from MSA file name)
        - Outgroup sequence id (default is none) 
    If we're filtering out non-eukaryotic species, the assumed format of the MSA file is stockholm. See details in `get_euk_seqs`
    """
    parser = argparse.ArgumentParser(description='This script pre-processes an MSA using a specified reference sequence')
    parser.add_argument('MSA', type=str, help='Path to MSA file')
    parser.add_argument('query_seq_id', type=str, help='ID of the sequence used as reference')
    parser.add_argument('--simul', action='store_true', help='Whether the MSA was simulated (i.e. not a PFAM family)')
    parser.add_argument('--filter_euks', action='store_true', help='Filter out non-eukaryotic sequences')
    parser.add_argument('--metadata', type = str, help = 'Path to file with metadata on (only the) eukaryotic sequences in the MSA')
    parser.add_argument('--outgroup_acc', type = str, help = 'Accession of a non-Eukaryotic sequence that we include for outgroup purposes')
    args = parser.parse_args()
    return args 
    
def get_euk_seqs(msa_file_path, metadata_file_path, outgroup_acc):
    """
    This function places eukaryotic sequences from the msa file into a dictionary seq_dicts, maxing out at MAX_SEQS sequences
    It assumes that the msa file is in stockholm format with #=GS lines that include the accessions of the sequences

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
            # or if it's the outgroup sequence
            if seq_id not in euk_ids and seq_id != outgroup_acc:
                continue
            else:
                seq_dict[seq_id] = seq.upper()
                count_sequences += 1
            # we can break if we've reached the maximum number of sequences
            if count_sequences > MAX_SEQS:
                break
    return seq_dict, euk_ids

def get_seqs(msa_file_path, sim):
    """
    This function processes the sequences from the MSA file of a specified format into a dictionary `seq_dict`
    without doing any filtering based on eukaryotic status.
    If the MSA is simulated, the file is in a format that we have to handle by creating a new temporary file. 
    In addition, the MSA in this case includes ancestral sequences that we want to exclude (those whose names don't begin with N).
    """
    if sim:
        with open(msa_file_path, 'r') as file_handle:
            with open("temp.txt", 'w') as temp_file:
                for i, line in enumerate(file_handle):
                    if i >= 17:
                        temp_file.write(line)
        msa_file_path = "temp.txt"
    format = "tab" if sim else "fasta"
    # Initialize dictionary that will be returned
    seq_dict = {} # sequence id -> sequence
    count_sequences = 0
    with open(msa_file_path, 'r') as file_handle:
        for record in SeqIO.parse(file_handle, format):
            if sim and record.id[0] != "N":
                continue
            count_sequences += 1
            seq_dict[record.id] = str(record.seq).upper()
            if count_sequences > MAX_SEQS:
                break
    if sim:
        remove("temp.txt")
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

def remove_seqs(seq_dict, outgroup_acc):
    """
    Remove sequences with too many remaining gaps (called after removing positions that are gaps in the query sequences)
    We retain the outgroup sequence even if it has many gaps
    """
    for k in list(seq_dict.keys()):
        if k == outgroup_acc:
            continue
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
    print(len(pos_idx), seq_ary.shape[1])
    return seq_ary[:, np.array(pos_idx)]

def remove_dupes(seq_ary, seq_names):
    """
    Remove duplicate sequences. 
    Query sequence should always be the first sequence in the array so shouldn't get removed even if it's a duplicate
    Outgroup sequence should be sufficiently different from eukaryotic sequences so that it doesn't get removed
    """
    seq_ary, idx = np.unique(seq_ary, axis = 0, return_index=True)
    seq_names = [seq_names[i] for i in idx]
    return seq_ary, seq_names

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
    # Get PFAM accession to use as a directory name for file saving/loading
    msa_file_path = args.MSA
    msa_name = path.splitext(path.basename(msa_file_path))[0]
    acc = msa_name.split("_")[0]
    if args.simul:
        processed_directory = f"data/simulations/processed/{acc}"
    else:
        processed_directory = f"data/real/processed/{acc}"
    if not path.exists(processed_directory):
        makedirs(processed_directory)
        
    # Save the mapping between characters and indices
    with open(f"{processed_directory}/aa_index.pkl", 'wb') as file_handle:
        pickle.dump(AA_INDEX, file_handle)

    # If we're filtering to eukaryotic sequences, we need a metadata file
    if args.filter_euks and args.metadata: # use the metadata file provided
        metadata_file_path = args.metadata_file
    elif args.filter_euks: # infer metadata file name from MSA file name
        metadata_file_path = f"data/Ding/metadata/{acc}_eukaryotes.tsv"

    # Load the sequences from the MSA file
    if args.filter_euks:
        seq_dict, euk_ids = get_euk_seqs(msa_file_path, metadata_file_path, args.outgroup_acc)
        # Save the mapping between labels and accessions
        with open(f"{processed_directory}/label_accession_mapping.pkl", 'wb') as file_handle:
            pickle.dump(euk_ids, file_handle)
    else:
        seq_dict, euk_ids = get_seqs(msa_file_path, args.simul), None

    # ensure that the query accession is in the MSA
    assert args.query_seq_id in seq_dict, f"Query accession {args.query_seq_id} not found in MSA"
    # ensure that the outgroup accession is in the MSA
    if args.outgroup_acc:
        assert args.outgroup_acc in seq_dict, f"Outgroup accession {args.outgroup_acc} not found in MSA"

    #### Preprocessing #####
    # Step 1: remove all positions that are gaps in the query sequences
    remove_gaps(seq_dict, args.query_seq_id) 

    # Step 2: Remove sequences with too many gaps
    remove_seqs(seq_dict, args.outgroup_acc)
    
    # Step 3: Convert to an integer numpy array
    seq_ary, seq_names = to_numpy(seq_dict, args.query_seq_id)
    
    # Step 4: remove positions where too many sequences have gaps 
    seq_ary = remove_sparse_positions(seq_ary)
    # Step 5: remove duplicate sequences
    seq_ary, seq_names = remove_dupes(seq_ary, seq_names)
    # At this point, we save to disk a character encoded version of our processed MSA
    aa = ["."] + AA # this is so that aa[0] = "." 
    with open(f"{processed_directory}/seq_msa_char.txt", "w") as f:
        for seq_id, seq in zip(seq_names, seq_ary.tolist()):
            decoded_seq = "".join([aa[i] for i in seq])
            f.write(f"{seq_id}\t{decoded_seq}\n")
    
    # We now drop the outgroup sequence, since the rest of the saved objects will be used to train VAEs
    # and we don't want the outgroup sequence to be included in the training data
    if args.outgroup_acc:
        seq_ary = np.delete(seq_ary, seq_names.index(args.outgroup_acc), axis = 0)
        seq_names.remove(args.outgroup_acc)

    # Step 6: get sequence weights
    seq_weight = weight_seqs(seq_ary)

    # Step 7: Convert to a binary (one-hot) encoding
    seq_ary_binary = one_hot_encode(seq_ary)

    #### Saving results ####
    # save the sequence labels
    with open(f"{processed_directory}/seq_names.pkl", 'wb') as file_handle:
        pickle.dump(seq_names, file_handle)
    # save the sequence weights
    with open(f"{processed_directory}/seq_weight.pkl", 'wb') as file_handle:
        pickle.dump(seq_weight, file_handle)
    # save integer encoded MSA
    with open(f"{processed_directory}/seq_msa.pkl", 'wb') as file_handle:
        pickle.dump(seq_ary, file_handle)
    # save one-hot encoded MSA
    with open(f"{processed_directory}/seq_msa_binary.pkl", 'wb') as file_handle:
        pickle.dump(seq_ary_binary, file_handle)

if __name__ == "__main__":
    main()