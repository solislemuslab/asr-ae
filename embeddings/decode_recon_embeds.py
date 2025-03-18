import sys
import os
import re
import argparse
from types import SimpleNamespace
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import pickle
from Bio import SeqIO
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilities.seq import invert_dict, filter_fasta, aa_to_int_from_path 
from utilities.model import load_model
from utilities.paths import get_directory
from utilities.tree import get_depths

def get_real_seqs(msa_path, anc_id, preserved_pos=None):
    sim_type = msa_path.split(os.sep)[1]
    format = {
        'potts': "fasta",
        'real' : "fasta",
        'coupled' : "phylip-relaxed",
        'independent' : "phylip-relaxed" 
    }[sim_type]
    real_seqs_dict = {}
    with open(msa_path, 'r') as msa:
        for record in SeqIO.parse(msa, format):
            if record.id[0] == "N":  # exclude leaf sequences
                continue
            seq = str(record.seq)
            if preserved_pos:
                seq = "".join([seq[pos] for pos in preserved_pos])
            real_seqs_dict[record.id] = seq
    # order true ancestral sequences according to the order of the reconstructed embeddings
    real_seqs = [real_seqs_dict[id] for id in anc_id]
    return real_seqs

def get_prior_seqs(model, n_seqs, index_aa):
    dim_latent_space = model.dim_latent_vars
    # sample from standard normal
    z = torch.randn(n_seqs, dim_latent_space)
    # decode to amino acid probabilities
    with torch.no_grad():
        log_p = model.decoder(z)
        p = torch.exp(log_p)
    p = p.numpy()
    p.shape
    # covert probablities into actual protein sequences by choosing the most likely amino acid at each position.
    max_prob_idx = np.argmax(p, -1)
    prior_seqs = []
    for i in range(n_seqs):
        seq = [index_aa[idx] for idx in max_prob_idx[i,:]]
        prior_seqs.append("".join(seq))
    return prior_seqs

def get_recon_ancseqs(recon_embeds, model, index_aa):
    # convert reconstructed embeddings to torch tensor
    mu = torch.tensor(recon_embeds.values , dtype=torch.float32)
    # now we decode
    with torch.no_grad():
      log_p = model.decoder(mu)
    # Now we covert probablities into actual protein sequences by choosing the most likely amino acid at each position.
    max_prob_idx = torch.argmax(log_p, -1)
    # Decode the integer `max_prob_idx` to character and display reconstructed ancestral sequences
    max_prob_idx = max_prob_idx.numpy()
    recon_seqs = []
    for i in range(len(max_prob_idx)):
        recon_seq = [index_aa[idx] for idx in max_prob_idx[i,:]]
        recon_seqs.append("".join(recon_seq))
    return recon_seqs

def get_modal_seq(data_path, index_aa):
    with open(f"{data_path}/seq_msa_int.pkl", 'rb') as file_handle:
        real_seq_leaves_int = pickle.load(file_handle)
    real_seq_leaves = []
    for i in range(real_seq_leaves_int.shape[0]):
        seq = [index_aa[idx] for idx in real_seq_leaves_int[i,:]]
        real_seq_leaves.append("".join(seq))
    maj_seq = []
    for i in range(len(real_seq_leaves[0])):
        col = [seq[i] for seq in real_seq_leaves]
        maj_seq.append(max(set(col), key=col.count))
    maj_seq = "".join(maj_seq)
    return maj_seq

def run_iqtree(data_path, tree_path, iqtree_dir, redo):
    with open(f"{data_path}/final_seq_names.txt") as file:
        final_seq_names = file.read().splitlines()
    # we have to create a new fasta file with only the sequences that ended up in our final tree
    filter_fasta(f"{data_path}/seq_msa_char.fasta", f"{data_path}/seq_msa_char.fasta", keep=final_seq_names)
    
    # Scale the tree
    # scaled_tree_path = f"{tree_path}_scaled{scaling_factor}"
    # with open(tree_path, 'r') as tree_file:
    #     tree_content = tree_file.read()
    # scaled_tree_content = re.sub(r'(\d+\.\d+)', lambda x: str(float(x.group(1)) * scaling_factor), tree_content)
    # with open(scaled_tree_path, 'w') as scaled_tree_file:
    #     scaled_tree_file.write(scaled_tree_content)
    
    # Run IQTree with scaled tree (only actually runs if analysis has not yet been done or redo is true)
    redo = " -redo" if redo else ""
    os.system(f"iqtree/bin/iqtree2 -s {data_path}/seq_msa_char.fasta -m LG -te {tree_path} -asr -quiet {redo} -pre {iqtree_dir}/results")

def get_iqtree_ancseqs(iqtree_dir, n_seq, anc_id):
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
    #print(f"Number of nodes for which IQTree has made reconstructed sequences: {len(iq_df_sk.index.unique())}")
    iq_seqs = []
    for k in anc_id:
        node_id = f"Node{int(k) - n_seq}"
        if node_id not in iq_df_sk.index:
            print(f"Reconstructions for {node_id} not found in IQTree output because it is the root node in the trimmed tree, "
                  "and for some reason, IQTree does not reconstruct the sequence at the root node.")
            iq_seqs.append(None)
            continue
        recon_seq_df = iq_df_sk.loc[node_id]
        recon_seq = "".join(recon_seq_df.State.values)
        iq_seqs.append(recon_seq)
    return iq_seqs

def evaluate_seqs(est_seqs, real_seqs):
    correct = 0
    total = 0
    for (est_seq, real_seq) in zip(est_seqs, real_seqs):
      if est_seq: # only evaluate non-missing reconstructions (iqtree)
        n_correct = sum([est_c == real_c for (est_c, real_c) in zip(est_seq, real_seq)])
        correct += n_correct
        total += len(real_seq)
    print(f"correct: {correct}")
    print(f"total: {total}")
    print(f"Percentage correct: {np.round(100*correct/total, 2)}")

def plot_error_vs_depth(est_seqs, real_seqs, depths):
    ham_errors = []
    for (est_seq, real_seq) in zip(est_seqs, real_seqs):
        if not est_seq: # handle missing reconstructions (iqtree)
            ham_errors.append(None) 
            continue
        ham_error = sum([est_c != real_c for (est_c, real_c) in zip(est_seq, real_seq)])/len(real_seq)
        ham_errors.append(ham_error)
    plt.scatter(depths, ham_errors)
    plt.xlabel("Node depth")
    plt.ylabel("Hamming reconstruction error")
    plt.title("Accuracy of reconstructed sequences vs. depth in tree")
    plt.show()

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
    print("model_name: ", config.model_name)
    print("plot: ", config.plot)
    print("redo_iqtree:", config.redo_iqtree)
    print("-" * 50)
    MSA_id, msa_path, data_path, model_name = config.MSA_id, config.msa_path, config.data_path, config.model_name

    # load mappling from integer to amino acid and vice versa
    aa_index = aa_to_int_from_path(data_path)
    index_aa = invert_dict(aa_index)

    # load the model
    model_dir = get_directory(data_path, MSA_id, "saved_models")
    model_path = os.path.join(model_dir, model_name)
    # to load the model, need to supply the sequence length, number of latent dims, and number of hidden units in each layer
    # first, sequence length, which is contained as substring of MSA_id for non-potts msas
    nl_match = re.search(r'nl(\d+)', MSA_id)
    if nl_match:
        preserved_pos = None
        nl = int(nl_match.group(1))
    else: # one way to identify sequence length for potts simulated msas 
        with open(f"{data_path}/pos_preserved.pkl", 'rb') as file:
            preserved_pos = pickle.load(file)
            nl = len(preserved_pos)
    # next, latent dimension
    ld = int(re.search(r'ld(\d+)', model_name).group(1))
    # finally, number of hidden units in each layer
    layers_match = re.search(r'layers(\d+(\-\d+)*)', model_name)
    num_hidden_units = [int(size) for size in layers_match.group(1).split('-')]
    model = load_model(model_path, nl, nc=21,
                           num_hidden_units=num_hidden_units, nlatent = ld)  
    
    # Get our reconstructed ancestral embeddings
    embeds_dir = get_directory(data_path, MSA_id, "embeddings", data_subfolder=True)
    embeds_path = os.path.join(embeds_dir,
                               model_name.replace(".pt", "_anc-embeddings.csv"))
    recon_embeds = pd.read_csv(embeds_path, index_col=0)
    n_anc = recon_embeds.shape[0]
    anc_id = [str(id) for id in recon_embeds.index]

    # Run IQTree
    iqtree_dir = get_directory(data_path, MSA_id, "iqtree", data_subfolder=True)
    os.makedirs(iqtree_dir, exist_ok=True)
    n_seq = int(msa_path.split("/")[-2])
    family = MSA_id.split("-")[0]
    tree_path = f"trees/fast_trees/{n_seq}/{family}.sim.fully_trim.tree"
    print("Running IQTree...")
    run_iqtree(data_path, tree_path, iqtree_dir, redo=config.redo_iqtree)

    # Retrieve sequences
    real_seqs = get_real_seqs(msa_path, anc_id, preserved_pos)
    # TODO: embed the real ancestral sequences and visually compare them with the reconstructed embeddings
    maj_seq = get_modal_seq(data_path, index_aa)
    prior_seqs = get_prior_seqs(model, n_anc, index_aa)
    recon_ancseqs = get_recon_ancseqs(recon_embeds, model, index_aa)
    iqtree_seqs = get_iqtree_ancseqs(iqtree_dir, n_seq, anc_id)

    # Evaluate accuracy of different approaches
    print("-" * 50)
    print("Evaluating modal sequence")
    #print(maj_seq)
    evaluate_seqs([maj_seq]*n_anc, real_seqs)
    print("-" * 50)
    print("Evaluating sequences sampled under the prior of trained VAE")
    evaluate_seqs(prior_seqs, real_seqs)
    print("-" * 50)    
    print("Evaluating decoded reconstructed embeddings (our approach)")
    evaluate_seqs(recon_ancseqs, real_seqs)
    print("-" * 50)
    print("Evaluating iqtree ancestral sequences")
    evaluate_seqs(iqtree_seqs, real_seqs)
    print("-" * 50)

    # Plot reconstruction accuracy as a function of distance from root
    depths = get_depths(tree_path)
    ordered_depths = [depths[f"Node{int(k) - n_seq}"] for k in anc_id]
    plot_error_vs_depth(recon_ancseqs, real_seqs, ordered_depths)
    plot_error_vs_depth(iqtree_seqs, real_seqs, ordered_depths)
    plot_error_vs_depth([maj_seq]*n_anc, real_seqs, ordered_depths)
    plot_error_vs_depth(prior_seqs, real_seqs, ordered_depths)

if __name__ == "__main__":
    main()