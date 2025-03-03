import os
import pickle
import sys 
from Bio import SeqIO
from ete3 import Tree
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilities import config

def get_directory(data_path, MSA_id, folder, data_subfolder = False):
    """
    Take data path and MSA id and return the directory where the data is stored.
    """
    if MSA_id[0:3] == "COG": # this is a simulated dataset
        sim_type = os.path.dirname(data_path).split("/")[1] #either coupled or independent
        num_seqs = os.path.dirname(data_path).split("/")[-1] 
        dir =  f"{folder}/{sim_type}/{num_seqs}/{MSA_id}"
    else:
        dir = f"{folder}/real/{MSA_id}"
    if data_subfolder:
        dir_list = dir.split("/")
        dir_list.insert(1, "data")
        dir = ("/").join(dir_list)
    return dir

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

def aa_to_int_from_file(data_path, MSA_id):
    """
    Load the amino acid to integer mapping from pickle file that's already been created.
    """
    if MSA_id == "pevae":    
            with open(f"{data_path}/LG_matrix.pkl", 'rb') as file_handle:
                LG_matrix = pickle.load(file_handle)
            amino_acids = LG_matrix['amino_acids']
            aa_index = aa_to_int(amino_acids, config.UNKNOWN)
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
    keys_to_delete = [s for s in config.UNKNOWN if s != unknown_symbol]
    for symbol in keys_to_delete:
        if symbol in aa_index:
            del aa_index[symbol]
    index_aa = {v: k for k, v in aa_index.items()}
    return index_aa  

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


def get_depths(tree_path, tree_format=1):
    """
    Given path to tree, return dictionary mapping internal node names to their depth, i.e. their distance to nearest leaf.
    """

    def compute_up(tree):
        """ 
        Calling this function creates an attribute called `dists_below` for every node (including leaves) in the rooted tree,
        which stores the distance from the node to all leaves below the node.
        """
        for node in tree.traverse("postorder"):
            if node.is_leaf():
                node.dists_below = { node.name: 0}
            else: 
                node.dists_below = {}
                for child in node.get_children():
                    for leaf, d in child.dists_below.items():
                        node.dists_below[leaf] = d + child.dist

    def compute_down(tree):
        """
        tree must have attribute `dists_below` computed for every node
        Key idea: suppose that I'm some arbitrary node. For any leaf that's not in my subtree, the distance from me to it is 
        just the distance from my parent to it plus the distance from me to my parent. Well, if we're traversing in a preorder, then 
        the distance from my parent to it will have already been calculated. Yay.
        """
        for node in tree.traverse("preorder"):
            node.dists_all = node.dists_below.copy()
            if node.up:
                for leaf, d in node.up.dists_all.items():
                    if leaf not in node.dists_all:
                        node.dists_all[leaf] = d + node.dist


    tree = Tree(tree_path, format=tree_format)
    compute_up(tree)
    compute_down(tree)
    internal_depths = {}
    for node in tree.traverse():
        if node.is_leaf():
            continue
        internal_depths[node.name] = min(node.dists_all.values())
    return internal_depths




