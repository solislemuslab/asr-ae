import pickle
from utilities import constants

def aa_to_int(main_aa_symbols, unknown_aa_symbols):
    """
    From a list of amino acids symbols (sorted to be in the desired order) 
    and a list of symbols representing ambiguous residues,
    generates a dictionary mapping amino acid symbols to integer indices.

    - main_aas: list of amino acids to include in the dictionary
    - unknown_aas: list of amino acids to map to 0, representing either gaps or other amino acids with ambiguity
    """
    aa_index = {}
    for i, aa in enumerate(main_aa_symbols):
        aa_index[aa] = i + 1
    for i, aa in enumerate(unknown_aa_symbols):
        aa_index[aa] = 0
    return aa_index

def aa_to_int_from_path(data_path):
    """
    Load the amino acid to integer mapping from the family folder (the one that has the processed MSA files).
    """
    with open(f"{data_path}/aa_index.pkl", 'rb') as file_handle:
        aa_index = pickle.load(file_handle)
    return aa_index

def invert_dict(aa_index, unknown_symbol = '-'):
    """
    Takes: dictionary mapping amino acid symbols to integer indices
    Returns: Inverse dictionary that maps integers to amino acid symbols.

    Many characters may get mapped to 0 in aa_index, so we have to choose one of these to map 0 to in our inverse dictionary.
    This is specified by the argument `unknown_symbol`.
    """
    aa_index = aa_index.copy()
    keys_to_delete = [s for s in constants.UNKNOWN if s != unknown_symbol]
    for symbol in keys_to_delete:
        if symbol in aa_index:
            del aa_index[symbol]
    index_aa = {v: k for k, v in aa_index.items()}
    return index_aa  









