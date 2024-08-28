import os

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

def to_fasta(f_in, f_out, keep = None):
    if keep:
        def process_line(line):
            id, seq = line.split()
            if id in keep:
                return f">{id}\n{seq}\n"
            else: 
                return ""
    else:
        def process_line(line):
            id, seq = line.split()
            return f">{id}\n{seq}\n"
    with open(f_in) as in_file, open(f_out, "w") as out_file:
        for line in in_file:
            processed_line = process_line(line)
            out_file.write(processed_line)

