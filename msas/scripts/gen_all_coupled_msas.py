# Drawing on https://github.com/songlab-cal/CherryML/blob/main/cherryml/simulation/_simulate_msas.py
import argparse
import random
import warnings
from typing import Dict, List, Optional
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from ete3 import Tree
import glob
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from utilities.io._rate_matrix import read_rate_matrix, read_probability_distribution

amino_acids = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

def sample(probability_distribution: np.array, size: tuple = None) -> int:
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


def simulate_msas(
    tree_dir: str,
    msa_length: int,
    scaling_factor: float,
    rate_het: float,
    Q_path: str,
    pi_path: str,
    amino_acids: List[str] = amino_acids[:],
    strategy: str = "all_transitions",
    random_seed: int = 770,
    tree_format: Optional[int] = 5,
    output_msa_dir: Optional[str] = None,
):
    """
    Simulate an MSA with co-evolving sites for every tree in tree_dir

    No site rate heterogeneity considered here for now.
    default Tree_format = 5: assumes that neither internal node names, nor support values, are present
    """
    # create output directory if it does not exist
    if not os.path.exists(output_msa_dir):
        os.makedirs(output_msa_dir)
    
    # read rates and equilibrium distribution
    Q_df = read_rate_matrix(Q_path)
    Q = Q_df.to_numpy()
    pi_df = read_probability_distribution(pi_path)
    pi = pi_df.to_numpy().flatten()

    # amino acids
    pairs_of_amino_acids = [
        aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids
    ]
    assert pairs_of_amino_acids == pi_df.index.tolist()
    assert pairs_of_amino_acids == Q_df.columns.tolist()

    # we will assume that adjacent sites are coupled
    if msa_length % 2 != 0:
        warnings.warn("`msa_length` is not even. Incrementing by 1.")
        msa_length += 1
    contacting_pairs = [(i, i+1) for i in range(0, msa_length, 2)]
    n_contacting_pairs = len(contacting_pairs)

    # get list of cleaned tree files (note that this assumes that the independent simulations have already been run, producing these cleaned trees)
    tree_files = glob.glob(os.path.join(tree_dir, '*.tree_cleaned'))

    # Simulate an MSA for each tree
    for tree_file in tree_files:
        print("Processing tree:", tree_file)
        tree = Tree(tree_file, format=tree_format)
        
        # counter for labelling internal nodes
        n_tips = len(tree.get_leaves())
        counter = n_tips + 1
        
        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # sample root state
        msa_int = {} # Dict[str, List[int]]
        tree.name = str(counter)
        root_states = list(sample(pi, (n_contacting_pairs,)))
        msa_int[tree.name] = root_states
        counter += 1

        # traverse the tree
        for node in tree.traverse("preorder"):
            if node.is_root():
                continue
            int_seq = []
            branch_length = node.dist * scaling_factor
            parent_state = msa_int[node.up.name]
            for i in range(n_contacting_pairs):
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

        # Now just map back the integer states to amino acids
        msa = {} # Dict[str, str]
        for node_name, int_seq in msa_int.items():
            str_seq = [""] * msa_length
            for i in range(n_contacting_pairs):
                aa_pair = pairs_of_amino_acids[int_seq[i]]
                (site_1, site_2) = contacting_pairs[i]
                str_seq[site_1] = aa_pair[0]
                str_seq[site_2] = aa_pair[1]
            msa[node_name] = "".join(str_seq)

        # convert MSA to list of SeqRecord objects
        seq_records = [SeqRecord(Seq(sequence), id=seq_id, description="") for seq_id, sequence in msa.items()]

        # write MSA to file
        fam_name = os.path.basename(tree_file).split('.')[0]
        output_file = f"{output_msa_dir}/{fam_name}-l{msa_length}-s{scaling_factor}-a{rate_het}_msa.dat"
        with open(output_file, 'w') as new:
            SeqIO.write(seq_records, new, "phylip-relaxed")

        # increment RNG seed
        random_seed += 1

if __name__ == "__main__":

    # Simulation parameters
    parser = argparse.ArgumentParser(
        description="Simulate an MSA with co-evolving sites.")
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

    # Directories of tree files and output MSAs
    tree_dirs = {
        '1250': '../trees/fast_trees/1250',
        '5000': '../trees/fast_trees/5000'
    }
    output_dirs = {
        '1250': 'coupling_sims/raw/1250',
        '5000': 'coupling_sims/raw/5000'
    }

    # Rate matrix and equilibrium distribution for simulation
    Q_path = "coupling_sims/coevolution.txt"
    pi_path = "coupling_sims/coevolution_stationary.txt"

    # Simulate MSAs for all trees with 1250 leaves and all trees with 5000 leaves
    for key in tree_dirs:
        simulate_msas(tree_dirs[key], msa_length, scaling_factor,
                      rate_het, Q_path, pi_path,
                      output_msa_dir=output_dirs[key])
