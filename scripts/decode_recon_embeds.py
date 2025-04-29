import sys
import os
import re
import argparse
from types import SimpleNamespace
import json
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
import torch
import pickle
from Bio import SeqIO
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilities.seq import aa_to_int_from_path
from utilities.vae import load_model
from utilities.utils import get_directory, parse_model_name
from utilities.tree import get_depths, run_fitch, run_iqtree

def get_real_seqs(msa_path, anc_id, aa_index, pos_preserved=None):
    format = "fasta"
    real_seqs_dict = {}
    with open(msa_path, 'r') as msa:
        for record in SeqIO.parse(msa, format):
            if record.id[0] == "N":  # exclude leaf sequences
                continue
            seq = str(record.seq)
            if pos_preserved:
                seq = "".join([seq[pos] for pos in pos_preserved])
            real_seqs_dict[record.id] = seq
    # order true ancestral sequences according to the order of the reconstructed embeddings
    real_seqs = [real_seqs_dict[id] for id in anc_id]
    # convert to integers for comparison with reconstructed sequences
    real_seqs_int = [[aa_index[aa] for aa in seq] for seq in real_seqs]
    return np.array(real_seqs_int)

def get_prior_seqs(model, n_anc):
    dim_latent_space = model.dim_latent_vars
    # sample from standard normal
    z = torch.randn(n_anc, dim_latent_space)
    # decode to amino acid probabilities
    with torch.no_grad():
        log_p = model.decoder(z)
    max_prob_idx = torch.argmax(log_p, -1)
    return max_prob_idx.numpy()

def get_recon_ancseqs(model, recon_embeds):
    # convert reconstructed embeddings to torch tensor
    mu = torch.tensor(recon_embeds.values , dtype=torch.float32)
    # now we decode
    with torch.no_grad():
      log_p = model.decoder(mu)
    max_prob_idx = torch.argmax(log_p, -1)
    return max_prob_idx.numpy()

def get_modal_seq(data_path, n_anc):
    with open(f"{data_path}/seq_msa_int.pkl", 'rb') as file_handle:
        real_seq_leaves= pickle.load(file_handle)
    mod_seq = []
    for i in range(real_seq_leaves.shape[1]):
        col = real_seq_leaves[:, i]
        mod_seq.append(np.bincount(col).argmax())
    return np.tile(np.array(mod_seq), (n_anc, 1))

def get_iqtree_ancseqs(iqtree_dir, anc_id, aa_index, n_seq):
    """
    Note that for some reason, IQTree does not reconstruct the sequence at the root node. Not sure why...
    Usually, our trees will be unrooted, but occassionally, the trimming of extremely short external branches 
    (for the software that fits a Brownian motion on the embeddings to reconstruct ancestral embeddings)
    results in a tree that is rooted. In this case, there will be one node (i.e. the root) for which we
    are able to obtain a reconstructed embedding that we can convert to a sequence, but for which IQ won't produce an ancestral sequence. 
    For now, this function just notifies the user of the lack of predictions from IQTree for this node.
    In the future, we can try to change this so that we evaluate IQTree and our approach on precisely the same set of nodes.
    """
    iq_df = pd.read_table(f'{iqtree_dir}/results.state', header=8)
    iq_df_sk = iq_df[["Node", "Site", "State"]]
    iq_df_sk = iq_df_sk.sort_values(by=["Node", "Site"])
    iq_df_sk.set_index("Node", inplace=True) 
    seq_length = iq_df_sk["Site"].max()
    placeholder = [-1] * seq_length # placeholder for missing reconstructions
    iq_seqs = []
    for id in anc_id:
        if id not in iq_df_sk.index:
            print(f"Reconstructions for Node {id} not found in IQTree output because it is the root node in the cleaned tree, "
                  "and for some reason, IQTree does not reconstruct the sequence at the root node.")
            iq_seqs.append(placeholder)
            continue
        recon_seq_df = iq_df_sk.loc[id]
        recon_seq = [aa_index[char] for char in recon_seq_df.State.values]
        iq_seqs.append(recon_seq)
    return np.array(iq_seqs)

def get_finch_ancseqs(recon_fitch_dict, anc_id, aa_index):
    finch_seqs = []
    for id in anc_id:
        assert id in recon_fitch_dict, f"Reconstruction for Node {id} not found in Fitch output"
        recon_seq = [aa_index[char] for char in recon_fitch_dict[id]]
        finch_seqs.append(recon_seq)
    return np.array(finch_seqs)

def get_ardca_ancseqs(ardca_dir, anc_id, aa_index):
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

def evaluate_seqs(est_seqs, real_seqs):
    assert est_seqs.shape == real_seqs.shape
    # filter out missing reconstructions
    mask = ~np.all(est_seqs == -1, axis=1)
    est_seqs, real_seqs = est_seqs[mask], real_seqs[mask]
    total = est_seqs.size
    correct = np.sum(est_seqs == real_seqs)
    print(f"correct: {correct}")
    print(f"total: {total}")
    print(f"Percentage correct: {np.round(100*correct/total, 2)}")

def plot_error_vs_depth(est_seqs, real_seqs, depths):
    ham_errors = []
    for (est_seq, real_seq) in zip(est_seqs, real_seqs):
        if all(est_seq == -1):  # handle missing reconstructions (iqtree)
            ham_errors.append(None)
            continue
        ham_errors.append(np.mean(est_seq != real_seq))
    
    # Scatter plot
    plt.scatter(depths, ham_errors, label="Hamming Error", alpha=0.5)
    
    # Add LOESS smooth
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


def plot_all_errors(all_est_seqs_dict, real_seqs, depths, output):
    """
    Plot the errors for all the sequences in all_est_seqs_dict.
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
                if name.startswith("model"):
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
    plt.savefig(output, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description='Decode \'ancestral\' embeddings into reconstructed ancestral sequences and evaluate accuracy')
    parser.add_argument('config_file', nargs='?', default='embeddings/config_decode.json', 
                    help='Path to configuration file specifying details, such as which family reconstructions are for, etc.')
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    print(f"Executing {sys.argv[0]} following {args.config_file}")
    print("-" * 50)
    print("MSA_id: ", config.MSA_id)
    print("msa_path: ", config.msa_path)
    print("data_path: ", config.data_path)
    print("model_name: ", config.model_names)
    print("plot: ", config.plot)
    print("redo_iqtree:", config.redo_iqtree)
    print("-" * 50)
    MSA_id, msa_path, data_path, model_names = config.MSA_id, config.msa_path, config.data_path, config.model_names

    # Get global variables
    n_seq = int(msa_path.split("/")[-2]) # number of sequences
    family = MSA_id.split("-")[0] # family name 
    tree_path = f"trees/fast_trees/{n_seq}/{family}.clean.tree" # path to tree
    # sequence length will be present in the MSA_id if it is not a Potts-simulated MSA 
    nl_match = re.search(r'l(\d+)', MSA_id)
    if nl_match:
        nl = int(nl_match.group(1))
        pos_preserved = None
    else:  
        with open(f"{data_path}/pos_preserved.pkl", 'rb') as file:
            pos_preserved = pickle.load(file) # positions preserved in processed MSA (only relevant for Potts)
        nl = len(pos_preserved)
    
    # load mappling from integer to amino acid and vice versa
    aa_index = aa_to_int_from_path(data_path)

    # location of reconstructed embeddings of ancestral sequences
    embeds_dir = get_directory(data_path, "embeddings", data_subfolder=True)
    
    # if model name is an empty list, then run with all models that have been fit to the given family
    model_dir = get_directory(data_path, "saved_models")
    if not model_names:
        model_names = [name for name in os.listdir(model_dir) if name.endswith(".pt")]
    model_dict = {} 
    for name in model_names:
        is_trans, ld, num_hidden_units, dim_aa_embed, one_hot = parse_model_name(name)
        model_path = os.path.join(model_dir, name)
        model = load_model(model_path, nl=nl, nc=21,
                            num_hidden_units=num_hidden_units, nlatent=ld,
                            one_hot=one_hot, dim_aa_embed=dim_aa_embed, trans=is_trans)
        embeds_path = os.path.join(embeds_dir,
                               name.replace(".pt", "_anc-embeddings.csv"))
        embeds = pd.read_csv(embeds_path, index_col=0)
        model_dict[name] = {
            "path": model_path,
            "model": model,
            "embeds": embeds
            }
    n_anc = model_dict[model_names[0]]["embeds"].shape[0] # number of ancestral sequences
    anc_id = [str(id) for id in model_dict[model_names[0]]["embeds"].index] # order of ancestral sequences
    
    # Run Fitch
    print("Running Fitch...")
    recon_fitch_dict = run_fitch(data_path, tree_path)
    # Run IQTree
    iqtree_dir = get_directory(data_path, "reconstructions/iqtree")
    os.makedirs(iqtree_dir, exist_ok=True)
    print("Running IQTree...")
    # TODO: use a different model and/or branch length optimization depending on the simulation type
    run_iqtree(data_path, tree_path, iqtree_dir, redo=config.redo_iqtree)
    # Retrieve real sequences
    real_seqs = get_real_seqs(msa_path, anc_id, aa_index, pos_preserved)
    # TODO: embed the real ancestral sequences and visually compare them with the reconstructed embeddings
    mod_seqs = get_modal_seq(data_path, n_anc)
    # Our approach
    for name in model_names:
        model = model_dict[name]["model"]
        recon_embeds = model_dict[name]["embeds"]
        model_dict[name]["prior_seqs"] = get_prior_seqs(model, n_anc)
        model_dict[name]["recon_seqs"] = get_recon_ancseqs(model, recon_embeds)
    # Other approaches
    iqtree_seqs = get_iqtree_ancseqs(iqtree_dir, anc_id, aa_index, n_seq=n_seq)
    finch_seqs = get_finch_ancseqs(recon_fitch_dict, anc_id, aa_index)
    ardca_dir = get_directory(data_path, "reconstructions/ardca")
    ardca_seqs = get_ardca_ancseqs(ardca_dir, anc_id, aa_index)

    # Evaluate accuracy of different approaches
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
    

    # Plot reconstruction accuracy as a function of distance from root
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
    # plot_error_vs_depth(recon_ancseqs, real_seqs, ordered_depths)
    plot_all_errors(all_est_seqs, real_seqs, ordered_depths, os.path.join(plot_dir, config.plot_name))
    
 
if __name__ == "__main__":
    main()