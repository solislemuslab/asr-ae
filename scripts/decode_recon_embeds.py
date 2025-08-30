import sys
import os
import json
import argparse
from typing import List
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
import torch
from torch import nn
import pickle
from Bio import SeqIO
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilities.seq import aa_to_int_from_path, invert_dict
from utilities.vae import load_model
from utilities.utils import get_directory, parse_model_name, get_real_internals
from utilities.tree import get_depths, run_fitch, run_iqtree


def get_prior_seqs(model: nn.Module, n_anc: int) -> NDArray[np.integer]:
    dim_latent_space = model.dim_latent_vars
    # sample from standard normal
    z = torch.randn(n_anc, dim_latent_space)
    # decode to amino acid probabilities
    with torch.no_grad():
        log_p = model.decoder(z)
    max_prob_idx = torch.argmax(log_p, -1)
    if log_p.shape[-1] == 20:
       # The model was trained without gaps, so we need to convert indices from 0-19 back to 1-20
       max_prob_idx = max_prob_idx + 1
    return max_prob_idx.numpy()

def get_recon_ancseqs(model: nn.Module, recon_embeds: pd.DataFrame, return_probs: bool=False) -> np.ndarray:
    """
    Decode reconstructed embeddings into sequences.
    - Returns np array with two axes (seq, position) of integers representing most like aa if return_probs=False (default).
    - Returns np array of probabilities with three axes (seq, position, aa) if return_probs=True.
    """
    mu = torch.tensor(recon_embeds.values , dtype=torch.float32)
    with torch.no_grad():
      log_p = model.decoder(mu)
    # Return distribution
    if return_probs:
        return log_p.exp().numpy()
    # Return most likely amino acid
    max_prob_idx = torch.argmax(log_p, -1)
    if log_p.shape[-1] == 20: # No gap character was included in model's data (one-hot-encoded MSA). We need to convert indices from 0-19 to 1-20
        max_prob_idx = max_prob_idx + 1
    return max_prob_idx.numpy()

def get_profile_leaves(data_path: str, n_anc: int, return_probs: bool=False) -> np.ndarray:
    """
    Get distribution of aa's (return_probs=True) or consensus aa (return_probs=False) 
    at each position in MSA of leaf sequences
    
    - If return_probs=False (default): Return numpy array with two axes (seq, position) of integers representing most common aa 
    - If return_probs=True: Return numpy array of probabilities with three axes (seq, position, aa) 
    
    In the latter case, the "aa" dimension will be 21 *even if the MSA has no gaps*.
    The "seq" dimension has length n_anc and just consists of repeating
    """
    with open(f"{data_path}/seq_msa_int.pkl", 'rb') as file_handle:
        real_seq_leaves = pickle.load(file_handle)
    nl = real_seq_leaves.shape[1]

    # Return consensus sequence
    if not return_probs:
        mod_seq = []
        for i in range(nl):
            col = real_seq_leaves[:, i]
            mod_seq.append(np.bincount(col).argmax())
        return np.tile(np.array(mod_seq, dtype=int), (n_anc, 1))
    
    # Return distribution 
    probs = np.zeros((nl, 21), dtype=np.float32)
    for i in range(nl):
        col = real_seq_leaves[:, i]
        counts = np.bincount(col, minlength=21).astype(np.float32)
        probs[i] = counts / counts.sum()
    return np.tile(probs, (n_anc, 1, 1))


def get_iqtree_ancseqs(iqtree_dir: str, aa_index: dict[str, int], 
                       anc_id: List[str], return_probs: bool=False, index_aa=None) -> np.ndarray:
    """
    Note that for some reason, IQTree does not reconstruct the sequence at the root node. Not sure why...
    

    Returns numpy array with two axes (seq, position) of integers representing most like aa if return_probs=False (default).
    Returns numpy array of probabilities with three axes (seq, position, aa) if return_probs=True.
    """
    
    message_template = """Reconstructions for Node {id} not found in IQTree output 
    because it is the root node in the cleaned tree, and for some reason, 
    IQTree does not reconstruct the sequence at the root node."""
    iq_df = pd.read_table(f'{iqtree_dir}/results.state', header=8)
    iq_df = iq_df.sort_values(by=["Node", "Site"])
    iq_df.set_index("Node", inplace=True) 
    seq_length = iq_df["Site"].max()
    
    # Return most likely amino acid
    if not return_probs:
        iq_df_sk = iq_df[["Node", "Site", "State"]]
        placeholder = [-1] * seq_length # placeholder for missing reconstructions
        iq_seqs = []
        for id in anc_id:
            if id not in iq_df_sk.index:
                print(message_template.format(id=id))
                iq_seqs.append(placeholder)
                continue
            recon_seq_df = iq_df_sk.loc[id]
            recon_seq = [aa_index[char] for char in recon_seq_df.State.values]
            iq_seqs.append(recon_seq)
        return np.array(iq_seqs)
    
    # Return distribution
    dists = []
    placeholder = np.full((seq_length, 20), -1, dtype=np.float32)
    for id in anc_id:
        if id not in iq_df.index:
            print(message_template.format(id=id))
            dists.append(placeholder)
            continue
        recon_seq_df = iq_df.loc[id]
        dist = recon_seq_df[[f"p_{index_aa[i]}" for i in range(1, 21)]].values
        dists.append(dist)
    return np.array(dists)

def get_finch_ancseqs(recon_fitch_dict: dict[str, List[str]], aa_index: dict[str, int], 
                      anc_id: List[str]) -> NDArray[np.integer]:
    finch_seqs = []
    for id in anc_id:
        assert id in recon_fitch_dict, f"Reconstruction for Node {id} not found in Fitch output"
        recon_seq = [aa_index[char] for char in recon_fitch_dict[id]]
        finch_seqs.append(recon_seq)
    return np.array(finch_seqs)

def get_ardca_ancseqs(ardca_dir: str, aa_index: dict[str, int], 
                      anc_id: List[str]) -> NDArray[np.integer]:
    fasta_path = os.path.join(ardca_dir, "reconstructed.fasta")
    recon_seqs_dict = {}
    with open(fasta_path, 'r') as msa:
        for record in SeqIO.parse(msa, "fasta"):
            recon_seqs_dict[record.id] = str(record.seq)
    # order true ancestral sequences according to the order of the reconstructed embeddings
    recon_seqs = [recon_seqs_dict[id] for id in anc_id]
    # convert to integers for comparison with reconstructed sequences
    recon_seqs_int = [[aa_index[aa] for aa in seq] for seq in recon_seqs]
    return np.array(recon_seqs_int)

def np_to_str(seq_ary: NDArray[np.integer] , index_aa: dict[int, str] ) -> List[str]:
    """
    Converts representation of reconstructed seqeunces from numpy array of integers to list of strings.
    """
    seqs = []
    for seq in seq_ary:
        seq_str = "".join([index_aa[i] for i in seq])
        seqs.append(seq_str)
    return seqs

def evaluate_seqs(est_seqs: NDArray[np.integer], real_seqs: NDArray[np.integer]) -> None:
    """
    Evaluate overall Hamming accuracy averaged over all reconstructed sequences. 
    """
    assert est_seqs.shape == real_seqs.shape
    # filter out missing reconstructions
    mask = ~np.all(est_seqs == -1, axis=1)
    est_seqs, real_seqs = est_seqs[mask], real_seqs[mask]
    total = est_seqs.size
    correct = np.sum(est_seqs == real_seqs)
    print(f"correct: {correct}")
    print(f"total: {total}")
    print(f"Percentage correct: {np.round(100*correct/total, 2)}")

def plot_error_vs_depth(est_seqs: NDArray[np.integer], real_seqs: NDArray[np.integer], depths: List[int]) -> List[float]:
    """
    Plot the Hamming reconstruction error as a function of the depth in the tree.

    Input:
    - est_seqs: estimated sequences (integer numpy array)
    - real_seqs: real sequences (integer numpy array)
    - depths: list of node depths (list of integers)
    Returns: list of Hamming errors for each reconstructed sequence
    """
    ham_errors = []
    for (est_seq, real_seq) in zip(est_seqs, real_seqs):
        if all(est_seq == -1):  # handle missing reconstructions (iqtree)
            ham_errors.append(None)
            continue
        ham_errors.append(np.mean(est_seq != real_seq))
    
    # Scatter plot
    plt.scatter(depths, ham_errors, label="Hamming Error", alpha=0.5)
    
    # Add LOWESS smooth
    valid_data = [(d, e) for d, e in zip(depths, ham_errors) if e is not None]
    if valid_data:
        valid_depths, valid_errors = zip(*valid_data)
        smoothed = lowess(valid_errors, valid_depths, frac=0.3)
        plt.plot(smoothed[:, 0], smoothed[:, 1], color="red", label="LOESS Smooth")
    
    # Plot settings
    plt.xlabel("Node depth")
    plt.ylabel("Hamming reconstruction error")
    plt.title("Accuracy of reconstructed sequences vs. depth in tree")
    plt.legend()
    plt.show()

    return ham_errors

def plot_all_errors(all_est_seqs_dict: NDArray[np.integer], real_seqs: NDArray[np.integer], 
                    depths: List[int], output: str=None) -> None:
    """
    Plot the errors for all the sequences in all_est_seqs_dict, saving the plot to output.
    """
    colors = {
        "modal": "blue",
        "fitch": "green",
        "iqtree": "orange",
        "ardca": "purple"
    }
    for name, est_seqs in all_est_seqs_dict.items():
        ham_errors = []
        for (est_seq, real_seq) in zip(est_seqs, real_seqs):
            if all(est_seq == -1):  # handle missing reconstructions (iqtree)
                ham_errors.append(None)
                continue
            ham_errors.append(np.mean(est_seq != real_seq))
            # Add LOESS smooth
            valid_data = [(d, e) for d, e in zip(depths, ham_errors) if e is not None]
        if valid_data:
            valid_depths, valid_errors = zip(*valid_data)
            try:
                smoothed = lowess(valid_errors, valid_depths, frac=0.3)
                # Set color and linestyle based on the name
                color = colors.get(name, "red")  # Default to red for our VAE-based approach
                linestyle = None
                if name.startswith("model") or name.startswith("ding"):
                    linestyle = "--"
                if name.endswith("prior"):
                    linestyle = "-."
                plt.plot(smoothed[:, 0], smoothed[:, 1], label=name, color=color, linestyle=linestyle)
            except Exception as e:
                print(f"Error while smoothing and plotting for {name}: {e}")
    # Plot settings
    plt.xlabel("Node depth")
    plt.ylabel("Hamming reconstruction error")
    plt.title("Accuracy of reconstructed sequences vs. depth in tree")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if output:
        plt.savefig(output, bbox_inches='tight')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Decode \'ancestral\' embeddings into reconstructed ancestral sequences and evaluate accuracy')
    parser.add_argument('config_file', nargs='?', default='config.json', 
                    help='Path to configuration file specifying details, such as which family reconstructions are for, etc.')
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    tree_path, msa_path, data_path,  = config["tree_path"], config["msa_path"], config["data_path"]
    model_names =  config["decode"]["model_names"]
    redo_iqtree = config["decode"]["redo_iqtree"]   
    plot = config["decode"]["plot_eval"]
    plot_name = config["decode"]["plot_name"]
    # if model name is an empty list, then run with all models that have been fit to the given family
    model_dir = get_directory(data_path, "saved_models")
    if not model_names:
        model_names = [name for name in os.listdir(model_dir) if name.endswith(".pt")]
    print(f"Executing {sys.argv[0]} following {args.config_file}")
    print("-" * 50)
    print("msa_path: ", config["msa_path"])
    print("evaluating models: ", model_names)
    print("-" * 50)

    ### Get some important global variables ####
    # The raw MSA may have more positions than the processed due to dropping of positions with excessive gaps 
    with open(f"{data_path}/pos_preserved.pkl", 'rb') as file:
        pos_preserved = pickle.load(file) 
    nl = len(pos_preserved)
    # load mappling from integer to amino acid and vice versa
    aa_index = aa_to_int_from_path(data_path)
    index_aa = invert_dict(aa_index, unknown_symbol='-')
    nc = len(index_aa) # 20 or 21, should equal the last dimension of one-hot encoded MSA input to the model, based on processing script process_msa.py
    processed_msa_path = os.path.join(data_path, "seq_msa_char.fasta")

    ### Read in all models and their embeddings ####
    embeds_dir = get_directory(data_path, "embeddings", data_subfolder=True)
    model_dict = {} # all models
    for name in model_names:
        # Load model
        ding_model = name.startswith("ding")
        is_trans, ld, num_hidden_units, dim_aa_embed, one_hot = parse_model_name(name)
        model_path = os.path.join(model_dir, name)
        model = load_model(model_path, nl=nl, nc=nc, ding_model=ding_model,
                            num_hidden_units=num_hidden_units, nlatent=ld,
                            one_hot=one_hot, dim_aa_embed=dim_aa_embed, trans=is_trans)
        model.eval()
        # Read in embedding dataframe and retain only the columns that are embedding dimensions
        embeds_path = os.path.join(embeds_dir,
                               name.replace(".pt", "_anc-embeddings.csv"))
        embeds = pd.read_csv(embeds_path, index_col="id")
        embeds = embeds.loc[:, embeds.columns.str.startswith("dim")]
        # Save model data
        model_dict[name] = {
            "path": model_path,
            "model": model,
            "embeds": embeds
            }
    n_anc = model_dict[model_names[0]]["embeds"].shape[0] # number of ancestors
    anc_id = model_dict[model_names[0]]["embeds"].index.tolist() #ancestor ids
    # check that embedding dataframes have the same order of ancestors
    for name, model_data in model_dict.items():
        if not anc_id == model_data["embeds"].index.tolist():
            print(f"Mismatch in ancestral orderings between embeddings from '{name}' and from '{model_names[0]}':")
            mismatched_index = next(
                (i for i, (a, b) in enumerate(zip(anc_id, model_data["embeds"].index)) if a != b),
                None
            )
            raise ValueError(
                f"Mismatch in ancestral orderings between embeddings from '{name}' and from '{model_names[0]}': "
                f"First mismatch at position {mismatched_index}: "
                f"Expected '{anc_id[mismatched_index]}', got '{model_data['embeds'].index[mismatched_index]}'."
            )
    

    ### Get reconstructed ancestral sequences from different approaches ####
    # Run Fitch
    print("Running Fitch...")
    _, recon_fitch_dict = run_fitch(processed_msa_path, tree_path)
    # Run IQTree
    iqtree_dir = get_directory(data_path, "reconstructions/iqtree")
    os.makedirs(iqtree_dir, exist_ok=True)
    print("Running IQTree...")
    # TODO: use a different model and/or branch length optimization depending on the simulation type
    run_iqtree(processed_msa_path, tree_path, iqtree_dir, redo=redo_iqtree)
    # Retrieve real internal sequences
    real_seqs, _ = get_real_internals(msa_path, aa_index, anc_id, pos_preserved)
    # Retrieve a sequence that is a consensus of the real leaf sequences
    mod_seqs = get_profile_leaves(data_path, n_anc)
    # Our approach
    for name in model_names:
        model = model_dict[name]["model"]
        recon_embeds = model_dict[name]["embeds"]
        model_dict[name]["prior_seqs"] = get_prior_seqs(model, n_anc)
        model_dict[name]["recon_seqs"] = get_recon_ancseqs(model, recon_embeds)
    # Other approaches
    iqtree_seqs = get_iqtree_ancseqs(iqtree_dir, aa_index, anc_id)
    finch_seqs = get_finch_ancseqs(recon_fitch_dict, aa_index, anc_id)
    ardca_dir = get_directory(data_path, "reconstructions/ardca")
    ardca_seqs = get_ardca_ancseqs(ardca_dir, aa_index, anc_id)



    ### Evaluate accuracy of different approaches ####
    print("-" * 50)
    print("Evaluating modal sequence")
    #print(maj_seq)
    evaluate_seqs(mod_seqs, real_seqs)
    print("-" * 50)
    for name in model_names:
        print(f"Evaluating sequences sampled under the prior of {name}")
        prior_spiky_seqs = model_dict[name]["prior_seqs"]
        evaluate_seqs(prior_spiky_seqs, real_seqs)
        print("-" * 50) 
        print(f"Evaluating decoded reconstructed embeddings of {name} (our approach)")
        recon_seqs = model_dict[name]["recon_seqs"]
        evaluate_seqs(recon_seqs, real_seqs)
        print("-" * 50)
    print("Evaluating Fitch reconstructed sequences")
    evaluate_seqs(finch_seqs, real_seqs)
    print("-" * 50)
    print("Evaluating iqtree ancestral sequences")
    print(iqtree_seqs.shape, " ", real_seqs.shape)
    evaluate_seqs(iqtree_seqs, real_seqs)
    print("-" * 50)
    print("Evaluating ArDCA reconstructed sequences")
    print(ardca_seqs.shape, " ", real_seqs.shape)
    evaluate_seqs(ardca_seqs, real_seqs)
    print("-" * 50)


    ### Plot reconstruction accuracy as a function of distance from root #####
    if plot:
        depths = get_depths(tree_path)
        ordered_depths = [depths[id] for id in anc_id]
        all_est_seqs = {
            "modal": mod_seqs,
            "fitch": finch_seqs,
            "iqtree": iqtree_seqs,
            "ardca": ardca_seqs,
        }
        for name in model_names:
            all_est_seqs[name] = model_dict[name]["recon_seqs"]
            all_est_seqs[f"{name}_prior"] = model_dict[name]["prior_seqs"]
        plot_dir = get_directory(data_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        # plot_error_vs_depth(mod_seqs, real_seqs, ordered_depths)
        # plot_error_vs_depth(prior_seqs, real_seqs, ordered_depths)
        # plot_error_vs_depth(finch_seqs, real_seqs, ordered_depths)
        # plot_error_vs_depth(iqtree_seqs, real_seqs, ordered_depths)
        # plot_error_vs_depth(ardca_seqs, real_seqs, ordered_depths)
        plot_all_errors(all_est_seqs, real_seqs, ordered_depths, os.path.join(plot_dir, plot_name))
        print("Evaluation plot saved to", os.path.join(plot_dir, plot_name))
    
    ### Append to csv with the ancestral embeddings the depths and the hamming errors #####
    for name, model_data in model_dict.items():
        recon_seqs = model_data["recon_seqs"]
        recon_embeds = model_data["embeds"]
        recon_embeds["recon_seqs"] = np_to_str(recon_seqs, index_aa)
        recon_embeds["depth"] = ordered_depths
        recon_embeds["ham_errors"] = [np.mean(seq != real_seq) for seq, real_seq in zip(recon_seqs, real_seqs)]
        recon_embeds.to_csv(os.path.join(embeds_dir, f"{name.replace('.pt', '_anc-embeddings.csv')}"), index=True)

if __name__ == "__main__":
    main()