import sys
import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import pickle
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from autoencoder.modules.model import load_model
from utilities.utils import get_directory, idx_to_aa, filter_fasta


def load_real_ancseqs(msa_path):
    real_seqs_dict = {}
    with open(msa_path, 'r') as msa:
        # skip the first line, which is Phylip header
        next(msa)
        # Read only ancestral sequences
        for line in msa: 
            id, seq = line.split()
            if id[0] != 'N':
                real_seqs_dict[id] = seq
    return real_seqs_dict

def get_prior_seqs(model, n_seqs, idx_to_aa_dict):
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
        seq = [idx_to_aa_dict[idx] for idx in max_prob_idx[i,:]]
        prior_seqs.append("".join(seq))
    return prior_seqs

def get_recon_ancseqs(recon_embeds, model, idx_to_aa_dict):
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
        recon_seq = [idx_to_aa_dict[idx] for idx in max_prob_idx[i,:]]
        recon_seqs.append("".join(recon_seq))
    return recon_seqs

def get_modal_seq(data_path, idx_to_aa_dict):
    with open(f"{data_path}/seq_msa_int.pkl", 'rb') as file_handle:
        real_seq_leaves_int = pickle.load(file_handle)
    real_seq_leaves = []
    for i in range(real_seq_leaves_int.shape[0]):
        seq = [idx_to_aa_dict[idx] for idx in real_seq_leaves_int[i,:]]
        real_seq_leaves.append("".join(seq))
    maj_seq = []
    for i in range(len(real_seq_leaves[0])):
        col = [seq[i] for seq in real_seq_leaves]
        maj_seq.append(max(set(col), key=col.count))
    maj_seq = "".join(maj_seq)
    return maj_seq

def run_iqtree(MSA_id, data_path, model_name, n_seq, iqtree_dir):
    with open(f"{data_path}/final_seq_names.txt") as file:
        final_seq_names = file.read().splitlines()
    # we have to create a new fasta file with only the sequences that ended up in our final tree
    filter_fasta(f"{data_path}/seq_msa_char.fasta", f"{data_path}/seq_msa_char.fasta", keep=final_seq_names)
    family = MSA_id.split("-")[0]
    tree_path = f"trees/fast_trees/{n_seq}/{family}.sim.trim.tree_revised"
    os.system(f"iqtree/bin/iqtree2 -s {data_path}/seq_msa_char.fasta -m LG -te {tree_path} -asr -redo -quiet -pre {iqtree_dir}/{model_name}")

def get_iqtree_ancseqs(iqtree_dir, model_name, n_seq, anc_id):
    iq_df = pd.read_table(f'{iqtree_dir}/{model_name}.state', header=8)
    iq_df_sk = iq_df[["Node", "Site", "State"]]
    iq_df_sk = iq_df_sk.sort_values(by=["Node", "Site"])
    iq_df_sk.set_index("Node", inplace=True)
    iq_seqs = []
    for k in anc_id:
        node_id = f"Node{int(k) - n_seq}"
        recon_seq_df = iq_df_sk.loc[node_id]
        recon_seq = "".join(recon_seq_df.State.values)
        iq_seqs.append(recon_seq)
    return iq_seqs

def evaluate_seqs(est_seqs, real_seqs):
    correct = 0
    total = 0
    acc = []
    for (est_seq, real_seq) in zip(est_seqs, real_seqs):
      n_correct = sum([est_c == real_c for (est_c, real_c) in zip(est_seq, real_seq)])
      correct += n_correct
      total += len(real_seq)
      acc.append(n_correct)
    print(f"correct: {correct}")
    print(f"total: {total}")
    print(f"Percentage correct: {np.round(100*correct/total, 2)}")

def main():
    name_script = sys.argv[0]
    name_json = sys.argv[1]
    print("-" * 50)
    print("Executing " + name_script + " following " + name_json, flush=True)
    print("-" * 50)
    # opening Json file
    json_file = open(name_json)
    data_json = json.load(json_file)
    MSA_id = data_json["MSA_id"]
    msa_path = data_json["msa_path"]
    data_path = data_json["data_path"]
    model_name = data_json["model_name"]
    plot = data_json["plot"]
    # print the information for training
    print("MSA_id: ", MSA_id)
    print("msa_path: ", msa_path)
    print("data_path: ", data_path)
    print("model_name: ", model_name)
    print("plot: ", plot)
    print("-" * 50)
    
    # load mappling from integer to amino acid and vice versa
    with open(f"{data_path}/aa_index.pkl", 'rb') as file_handle:
        aa_index = pickle.load(file_handle)
    idx_to_aa_dict = idx_to_aa(aa_index)

    # load the real ancestral sequences     
    real_seqs_dict = load_real_ancseqs(msa_path)
    
    # load the model
    nc = 21
    model_dir = get_directory(data_path, MSA_id, "saved_models")
    model_path = os.path.join(model_dir, model_name)
    nl = int(MSA_id.split("-")[1][1:])
    ld = int(re.search(r'ld(\d+)', model_name).group(1))
    model = load_model(model_path, nl, nc, nlatent = ld)

    # get our reconstructed ancestral embeddings
    embeds_dir = get_directory(data_path, MSA_id, "embeddings", data_subfolder=True)
    embeds_path = os.path.join(embeds_dir,
                               model_name.replace(".pt", "_anc-embeddings.csv"))
    recon_embeds = pd.read_csv(embeds_path, index_col=0)
    n_anc = recon_embeds.shape[0]
    anc_id = [str(id) for id in recon_embeds.index]

    # order true ancestral sequences according to the order of the reconstructed embeddings
    real_seqs = [real_seqs_dict[id] for id in anc_id]

    # TODO: embed the real ancestral sequences and visually compare them with the reconstructed embeddings

    # evaluate as baseline the accuracy of the modal sequence
    maj_seq = get_modal_seq(data_path, idx_to_aa_dict)
    print("Evaluating modal sequence")
    print(maj_seq)
    evaluate_seqs([maj_seq]*n_anc, real_seqs)
    print("-" * 50)

    # evaluate draws from the prior and compare them with the real ancestral sequences
    prior_seqs = get_prior_seqs(model, n_anc, idx_to_aa_dict)
    print("Evaluating prior samples")
    evaluate_seqs(prior_seqs, real_seqs)
    print("-" * 50)

    # evaluate the reconstructed embeddings and compare them with the real ancestral sequences
    recon_ancseqs = get_recon_ancseqs(recon_embeds, model, idx_to_aa_dict)
    print("Evaluating reconstructed embeddings")
    evaluate_seqs(recon_ancseqs, real_seqs)
    print("-" * 50)

    #run iqtree ancestral sequence reconstruction and evaluate
    print("Evaluating iqtree ancestral sequences")
    n_seq = int(msa_path.split("/")[-2])
    iqtree_dir = get_directory(data_path, MSA_id, "iqtree", data_subfolder=True)
    os.makedirs(iqtree_dir, exist_ok=True)
    run_iqtree(MSA_id, data_path, model_name, n_seq, iqtree_dir)
    iqtree_seqs = get_iqtree_ancseqs(iqtree_dir, model_name, n_seq, anc_id)
    evaluate_seqs(iqtree_seqs, real_seqs)
    print("-" * 50)


if __name__ == "__main__":
    main()