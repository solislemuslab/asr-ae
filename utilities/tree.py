import os
from ete3 import Tree
from Bio import SeqIO
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


def _fitch1(profile_seq1, profile_seq2):
    profile_seq = []
    for prof1, prof2 in zip(profile_seq1, profile_seq2):
        profile = prof1 & prof2 #intersection
        if not profile: #if intersection empty
            profile = prof1 | prof2 #union
        profile_seq.append(profile)
    return profile_seq


def _fitch2(profile_seq, parent_seq):
    seq = []
    for prof, char in zip(profile_seq, parent_seq):
        if char in prof:
            seq.append(char) # use parent's state
        else:
            seq.append(list(prof)[0]) # choose randomly from child's profile
    return seq

def run_fitch(data_path, tree_path):
    """
    Run Fitch algorithm to infer ancestral sequences. 
    Returns dictionary mapping internal node names to strings (reconstructed sequences)
    """
    tree = Tree(tree_path, format=1)
    # Because the tree is unrooted, but Ete3 reads it as rooted, there is a polytomy at the root
    # We can convert the polytomy at the root by creating an arbitrary dicotomic structure 
    # This is necessary for the fitch algorithm to work, as it requires a rooted binary tree
    # Alternatively, we can midpoint root.
    # I'm not sure if the ASR will be the same in both cases--my assumption is that the output of Fitch algorithm is sensitive to where root is placed.
    tree.resolve_polytomy(recursive=False) 
    profile_seqs = {}
    # get states at leaves from msa
    msa_path = os.path.join(data_path, "seq_msa_char.fasta")
    with open(msa_path, 'r') as file_handle:
        for record in SeqIO.parse(file_handle, "fasta"):
            profile_seqs[record.id] = [{char} for char in str(record.seq)]
    # traverse in postorder to get profiles
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            continue
        children_profile_seqs = [profile_seqs[child.name] for child in node.get_children()]
        assert len(children_profile_seqs) == 2, f"Tree is not binary {[child.name for child in node.get_children()]}"
        profile_seqs[node.name] = _fitch1(children_profile_seqs[0], children_profile_seqs[1])
    # make choices for the root
    root_reconstruction = [] 
    for (i, prof) in enumerate(profile_seqs[tree.name]):
        prof = list(prof)
        chosen = prof[0]
        if len(prof) > 1:
            print(f"Ambiguous state at root position {i}: {prof}. Choosing {chosen}")
        root_reconstruction.append(chosen)
    # traverse in preorder
    recon_seqs = {}
    recon_seqs[tree.name] = "".join(root_reconstruction)
    for node in tree.traverse("preorder"):
        if node.is_root() or node.is_leaf():
            continue
        parent = node.up
        recon_seqs[node.name] = _fitch2(profile_seqs[node.name], recon_seqs[parent.name])
    return recon_seqs

def run_iqtree(data_path, tree_path, iqtree_dir, 
               model="LG+G", optimize_branch_lengths=False, redo=False):
    # If you want to scale the tree:
    # scaled_tree_path = f"{tree_path}_scaled{scaling_factor}"
    # with open(tree_path, 'r') as tree_file:
    #     tree_content = tree_file.read()
    # scaled_tree_content = re.sub(r'(\d+\.\d+)', lambda x: str(float(x.group(1)) * scaling_factor), tree_content)
    # with open(scaled_tree_path, 'w') as scaled_tree_file:
    #     scaled_tree_file.write(scaled_tree_content)
    
    # Run IQTree if analysis has not yet been done or redo is true
    # TODO: make iqtree also consider gaps for reconstructed sequences
    # TODO: use model search instead of assuming LG model when run on Potts-simulated data
    # blfix is used to fix branch lengths
    redo_flag = " -redo" if redo else ""
    blfix_flag = " -blfix" if not optimize_branch_lengths else ""
    os.system(f"iqtree/bin/iqtree2 -s {data_path}/seq_msa_char.fasta -m {model} \
               -te {tree_path} -asr -quiet {redo_flag} {blfix_flag} -pre {iqtree_dir}/results")
    
    # Now we need to get the mapping since IQ-TREE will rename the internal nodes

    # Load original tree with correct internal node names
    # original_tree = Tree(tree_path, format=1)

    # # Load IQ-TREE output tree with renamed internal nodes
    # iqtree_tree = Tree(f"{iqtree_dir}/results.treefile", format=1)

    # # mapping = {}
    # # for node_iq, node_og in zip(iqtree_tree.traverse(), original_tree.traverse()):
    # #     if not node_iq.is_leaf():
    # #         print(node_iq.name)
    # #         assert not node_og.is_leaf()
    # #         mapping[node_og.name] = node_iq.name
    # # return mapping

def run_autoasr():
    pass 