# Code adapted from https://github.com/songlab-cal/CherryML/blob/main/cherryml/simulation/_simulate_msas.py
import argparse
import random
import warnings
from typing import Dict, List
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from ete3 import Tree
import glob
import sys
import os
import re 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilities import constants
from utilities.rate_matrix import (
    read_params_from_PAML, 
    read_rate_matrix_from_csv, 
    read_probability_distribution_from_csv,
    compute_scale
)

def sample(probability_distribution: np.array, size: tuple = None) -> int:
    probability_distribution = probability_distribution / np.sum(probability_distribution)
    return np.random.choice(
        range(len(probability_distribution)), size=size, p=probability_distribution
    )

def sample_transition(
        starting_state: int,
        rate_matrix: np.array,
        elapsed_time: float,
        strategy: str,
) -> int:
    """
    Sample the ending state of the Markov chain.

    Args:
        strategy: Either "all_transitions" or "chain_jump".
            The "all transitions" strategy simulates all transitions of the chain 
            during the elapsed time by simulating exponentials until their sum exceeds elapsed_time
            The "chain_jump" strategy directly computes the transition probability distribution with 
            the matrix exponential and then simulates from this distribution.
    """

    if strategy == "all_transitions":
        n = rate_matrix.shape[0]
        cur_state = starting_state
        cur_time = 0
        while True:
            wait_time = np.random.exponential(
                1.0 / -rate_matrix[cur_state, cur_state]
            )
            cur_time += wait_time
            if cur_time >= elapsed_time:
                # end of process
                return cur_state
            # update state
            weights = list(
                rate_matrix[cur_state, :cur_state]
            ) + list(rate_matrix[cur_state, (cur_state + 1):])
            assert len(weights) == n - 1
            new_state = random.choices(
                population=range(n - 1), weights=weights, k=1
            )[0]
            # new_state is now in [0, n-2], map it back to [0, n-1]
            if new_state >= cur_state:
                new_state += 1
            cur_state = new_state
    elif strategy == "chain_jump":
        raise NotImplementedError
    else:
        raise Exception(f"Unknown strategy: {strategy}")

def simulate_msa(tree_file, Q, pi, seq_len, 
                 tree_format=2, strategy="all_transitions") -> Dict[str, List[int]]:
    """
    Simulates MSA on a given tree file
    NOTE: Based on tree_format, default is Newick file with all branches + leaf names + internal supports
    NOTE: seq_len is the number of pairs of contacting sites in the case that we are using Cherry model
    NOTE: No site rate heterogeneity considered here for now.
    """
    print("Simulating MSA on", tree_file)
    # read tree
    tree = Tree(tree_file, format=tree_format)
    # counter for labelling internal nodes
    n_tips = len(tree.get_leaves())
    counter = n_tips + 1
    # sample root state
    msa_int = {} # Dict[str, List[int]]
    tree.name = str(counter)
    root_states = list(sample(pi, (seq_len,)))
    msa_int[tree.name] = root_states
    counter += 1
    # traverse the tree
    for node in tree.traverse("preorder"):
        if node.is_root():
            continue
        int_seq = []
        branch_length = node.dist * scaling_factor
        parent_state = msa_int[node.up.name]
        for i in range(seq_len):
            parent_state_i = parent_state[i]
            child_state_i = sample_transition(
                parent_state_i, Q, branch_length, strategy
            )
            int_seq.append(child_state_i)
        if not node.is_leaf():
            node.name = str(counter)
            counter += 1
            msa_int[node.name] = int_seq
        else:
            msa_int[node.name] = int_seq
    return msa_int

def convert_to_char(msa_int, state_symbols):
    """
    Note this works for both independent and coupled simulations
    """
    msa = {} # Dict[str, str]
    for node_name, int_seq in msa_int.items():
        str_seq = []
        for idx in int_seq:
            state = state_symbols[idx]
            str_seq.append(state)
        msa[node_name] = "".join(str_seq)
    return msa

if __name__ == "__main__":

    # Simulation parameters
    parser = argparse.ArgumentParser(
        description="Simulate MSAs for all trees")
    parser.add_argument("-type", type=str,
                        default="independent", help="Type of simulation (coupled or independent)")
    parser.add_argument("-l", "--length", type=int,
                        default=100, help="Length of the MSA")
    parser.add_argument("-s", "--scaling_factor", type=float,
                        default=1.0, help="Branch length scaling factor")
    parser.add_argument("-a", "--gamma_rate_heterogeneity", type=float,
                        default=None, help="Gamma rate heterogeneity (default: no heterogeneity)")
    args = parser.parse_args()

    msa_length = args.length
    scaling_factor = args.scaling_factor
    rate_het = args.gamma_rate_heterogeneity
    type = args.type

    # Directories of tree files and output MSAs
    tree_dirs = {
        '1250': 'trees/fast_trees/1250',
        '5000': 'trees/fast_trees/5000'
    }
    output_dirs = {
        '1250': f'msas/{type}/raw/1250',
        '5000': f'msas/{type}/raw/5000'
    }
    
    # Get seq_len, which is half of msa_length in the case of Cherry model
    if type == "coupled" and msa_length % 2 != 0:
        warnings.warn("`msa_length` is not even. Incrementing by 1.")
        msa_length += 1
    if type == "coupled":
        seq_len = msa_length // 2
    elif type == "independent":
        seq_len = msa_length
    
    # Get parameters and states
    if type == "coupled":    
        Q_df = read_rate_matrix_from_csv("msas/coupled/coevolution.txt")
        pi_df = read_probability_distribution_from_csv("msas/coupled/coevolution_stationary.txt")
        states = pi_df.index.tolist() # these are pairs of amino acid characters
        assert states == Q_df.columns.tolist()
        Q = Q_df.to_numpy()
        pi = pi_df.to_numpy().flatten()
    elif type == "independent":
        S, pi = read_params_from_PAML("msas/independent/lg_LG.PAML.txt")
        Q = S @ np.diag(pi)
        np.fill_diagonal(Q, -Q.sum(axis=1))
        states = constants.AA
    assert np.allclose(Q @ np.ones_like(pi), 0, atol=1e-6)
    assert np.allclose(pi @ Q, 0, atol=1e-6)
    print("Diagonal of Q:", np.diag(Q)[:10])
    print(compute_scale(Q, pi))
    sys.exit(0)
    
    # set seed
    random.seed(770)
    np.random.seed(770)

    # Simulate MSAs for all trees in each directory in tree_dirs
    for n_seq in tree_dirs:

        # create output directory for trees with x number of leaves 
        output_dir = output_dirs[n_seq]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # get list of cleaned tree files (note that this assumes that the independent simulations have already been run, producing these cleaned trees)
        tree_files = glob.glob(os.path.join(tree_dirs[n_seq], '*.sim.trim.tree'))

        # Simulate MSAs on all tree_files
        for tree_file in tree_files:
            # clean tree file by removing any negative branch lengths if necessary
            if n_seq == "5000":
                with open(tree_file, 'r') as file:
                    tree_content = file.read()
                    tree_content = re.sub(r'-\d+\.\d+', '0.0', tree_content)
                with open(tree_file, 'w') as file:
                    file.write(tree_content)
            # simulate MSA on tree_file
            msa_int = simulate_msa(tree_file, Q, pi, seq_len)
            # convert to characters
            msa = convert_to_char(msa_int, states)
            # convert MSA to list of SeqRecord objects and write to file
            seq_records = [SeqRecord(Seq(sequence), id=seq_id, description="") for seq_id, sequence in msa.items()]
            fam_name = os.path.basename(tree_file).split('.')[0]
            output_file = f"{output_dir}/{fam_name}-l{msa_length}-s{scaling_factor}-a{rate_het}_msa.phy"
            with open(output_file, 'w') as new:
                SeqIO.write(seq_records, new, "phylip-relaxed")