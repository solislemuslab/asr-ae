import pickle
from Bio import SeqIO
from utilities import constants

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

def aa_to_int(main_aa_symbols, unknown_aa_symbols):
    """
    From a list of amino acids symbols (sorted to be in the desired order) 
    and a list of symbols representing ambiguous residues,
    generates a dictionary mapping amino acid symbols to integer indices.

    main_aas: list of amino acids to include in the dictionary
    unknown_aas: list of amino acids to map to 0, representing either gaps or other amino acids with ambiguity
    """
    aa_index = {}
    for i, aa in enumerate(main_aa_symbols):
        aa_index[aa] = i + 1
    for i, aa in enumerate(unknown_aa_symbols):
        aa_index[aa] = 0
    return aa_index

def aa_to_int_from_path(data_path, is_pevae=False):
    """
    Load the amino acid to integer mapping from the family folder (the one that has the processed MSA files).
    """
    if is_pevae:    
            with open(f"{data_path}/LG_matrix.pkl", 'rb') as file_handle:
                LG_matrix = pickle.load(file_handle)
            amino_acids = LG_matrix['amino_acids']
            aa_index = aa_to_int(amino_acids, constants.UNKNOWN)
    else:
        with open(f"{data_path}/aa_index.pkl", 'rb') as file_handle:
            aa_index = pickle.load(file_handle)
    return aa_index

def invert_dict(aa_index, unknown_symbol = '.'):
    """
    Takes: dictionary mapping amino acid symbols to integer indices
    Returns: Inverse dictionary that maps integers to amino acid symbols.

    Many characters may get mapped to 0 in aa_index, so we have to choose one of these to map 0 to in our inverse dictionary.
    This is specified by the argument `unknown_symbol`.
    The function makes lasting changes to the input dictionary aa_index passed in so that after function call, only `unknown_symbol` is mapped to 0.
    """
    keys_to_delete = [s for s in constants.UNKNOWN if s != unknown_symbol]
    for symbol in keys_to_delete:
        if symbol in aa_index:
            del aa_index[symbol]
    index_aa = {v: k for k, v in aa_index.items()}
    return index_aa  









