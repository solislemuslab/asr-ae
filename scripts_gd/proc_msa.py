# code adapted from PEVAE paper

import argparse
import pickle
import sys
from os import path
import numpy as np
from sys import exit
import csv
import pandas as pd

# Parse command line arguments (path to MSA file, reference (aka query) sequence id, and optional path to metadata file)
parser = argparse.ArgumentParser(description='This script pre-processes an MSA using a specified reference sequence')
parser.add_argument('MSA', type=str, help='Path to MSA file')
parser.add_argument('query_seq_id', type=str, help='ID of the sequence used as reference')
parser.add_argument('outgroup_acc', type = str, help = 'Accession of a non-Eukaryotic sequence that we include for outgroup purposes')
parser.add_argument('--metadata', type = str, help = 'Path to file with metadata on (only the) eukaryotic sequences in the MSA')

args = parser.parse_args()

msa_file_path = args.MSA
msa_name = path.splitext(path.basename(msa_file_path))[0]
pfam_acc = msa_name.split("_")[0]
query_seq_id = args.query_seq_id
outgroup_acc = args.outgroup_acc
if args.metadata:
    metadata_file_path = args.metadata_file
else: # infer metadata file name from MSA file name
    metadata_file_path = f"data/pevae_real/{pfam_acc}_eukaryotes.tsv"

# load metadata 
mdata = pd.read_csv(metadata_file_path, delimiter='\t', index_col=False)
euk_accs = list(mdata.accession) # list of all eukaryotic sequence accessions
euk_accs.append(outgroup_acc) # include one non-Eukaryotic sequence as an out-of-group


## the following for loop extracts all sequences from eukaryotic species into a dictionary seq_dicts, maxing out at 50_000 sequences
## Note that we've edited the raw MSA file to put both the query and the outgroup sequences at the top in order to ensure         they're among the first of the 50,000 sequences
# We have a list of the accessions for these sequences (euk_accs) and we only want to save a sequence if its accession appears in this list.Unfortunately, in the part of the file where the actual sequences appear, they are labeled with something that's not the same thing as their accession. The code in the block "if line[0:4] == "#=GS":" adds to euk_ids only the labels that correspond to accessions that are in our list of eukaryote sequence accessions so that later in the for loop, we are able to save only the sequences whose labels are in euk_ids.
euk_ids = {} # sequence label -> sequence accession
seq_dict = {} # sequence label -> sequence
count_labels = 0 
count_sequences = 0
with open(msa_file_path, 'r') as file_handle:
    for line in file_handle:
        if line[0:4] == "#=GS" and count_labels <= 50_000:
            parts = line.split()
            acc, label = parts[-1].split(".")[0], parts[1]
            if acc in euk_accs:
                euk_ids[label] = acc
                count_labels += 1
                print(count_labels)
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
            print(count_sequences)
        if count_sequences > 50_000:  
            break

# Save the mapping between labels and accessions (i.e. what we've called euk_ids) for future reference
with open("./output/" + f"{msa_name}/label_accession_mapping.pkl", 'wb') as file_handle:
    pickle.dump(euk_ids, file_handle)

# Now we do some pre-processing of this MSA that contains only eukaryotic species
## Step 1: Remove all positions that are gaps in the query sequences
query_seq = seq_dict[query_seq_id] ## with gaps
idx = [ s == "-" or s == "." for s in query_seq]
for k in seq_dict.keys():
    seq_dict[k] = [seq_dict[k][i] for i in range(len(seq_dict[k])) if idx[i] == False]
query_seq = seq_dict[query_seq_id] ## without gaps

## Step 2. Remove sequences with too many gaps
len_query_seq = len(query_seq)
seq_id = list(seq_dict.keys())
num_gaps = []
for k in seq_id:
    num_gaps.append(seq_dict[k].count("-") + seq_dict[k].count("."))
    if seq_dict[k].count("-") + seq_dict[k].count(".") > 10 and euk_ids[k] != outgroup_acc:
        seq_dict.pop(k)
        
## convert aa type into num 0-20
aa = ['R', 'H', 'K',
      'D', 'E',
      'S', 'T', 'N', 'Q',
      'C', 'G', 'P',
      'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
aa_index = {}
aa_index['-'] = 0
aa_index['.'] = 0
# There are some B's in pf00041_full,
# Apparently, it respresents either D or N. We will instead treat it as missing for now
aa_index['B'] = 0 

i = 1
for a in aa:
    aa_index[a] = i
    i += 1
with open("./output/" + f"{msa_name}/aa_index.pkl", 'wb') as file_handle:
    pickle.dump(aa_index, file_handle)
    
seq_ary = [] # Integer encoded array representing the processed MSA
keys_list = [] # Sequence labels
for k in seq_dict.keys():
    if seq_dict[k].count('X') > 0 or seq_dict[k].count('Z') > 0:
        continue    
    seq_ary.append([aa_index[s] for s in seq_dict[k]])
    keys_list.append(k)    
seq_ary = np.array(seq_ary)

with open("./output/" + f"{msa_name}/keys_list.pkl", 'wb') as file_handle:
    pickle.dump(keys_list, file_handle)

## Step 3. remove positions where too many sequences have gaps
pos_idx = []
for i in range(seq_ary.shape[1]):
    if np.sum(seq_ary[:,i] == 0) <= seq_ary.shape[0]*0.2:
        pos_idx.append(i)


## Save the processed array
seq_ary = seq_ary[:, np.array(pos_idx)]
with open("./output/" + f"{msa_name}/seq_msa.pkl", 'wb') as file_handle:
    pickle.dump(seq_ary, file_handle)

# Save a character encoded array representing the processed MSA
aa = ["."] + aa
with open("./output/" + f"{msa_name}/seq_msa_char.txt", "w") as f:
    for seq_id, seq in zip(keys_list, seq_ary.tolist()):
        decoded_seq = "".join([aa[i] for i in seq])
        f.write(f"{seq_id}\t{decoded_seq}\n")
    
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
with open("./output/" + f"{msa_name}/seq_weight.pkl", 'wb') as file_handle:
    pickle.dump(seq_weight, file_handle)

## Save a binary encoded array representing the processed MSA
K = 21 ## num of classes of aa
D = np.identity(K)
num_seq = seq_ary.shape[0]
len_seq_ary = seq_ary.shape[1]
seq_ary_binary = np.zeros((num_seq, len_seq_ary, K)) # Binary encoded array representing the processed MSA
for i in range(num_seq):
    seq_ary_binary[i,:,:] = D[seq_ary[i]]

with open("./output/" + f"{msa_name}/seq_msa_binary.pkl", 'wb') as file_handle:
    pickle.dump(seq_ary_binary, file_handle)
