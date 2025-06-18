import argparse
import os
import glob
import numpy as np
from ete3 import Tree
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import torch
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from autoencoder.model import VAE
from utilities.seq import aa_to_int_from_path, invert_dict
from utilities.utils import parse_model_name
from utilities.vae import load_model

def simulate_brownian_on_tree(tree_file, latent_dim, scale=1.0, tree_format=1):
    tree = Tree(tree_file, format=tree_format)
    latent_vectors = {}
    # Simulate root
    root_vec = np.zeros(shape=(latent_dim,))
    latent_vectors[tree.name] = root_vec
    for node in tree.traverse("preorder"):
        if node.is_root():
            continue
        parent_vec = latent_vectors[node.up.name]
        branch_length = node.dist * scale
        child_vec = parent_vec + np.random.normal(0, np.sqrt(branch_length), size=(latent_dim,))
        latent_vectors[node.name] = child_vec
    return latent_vectors

def decode_latents_to_seqs(latent_vectors, vae, idx_to_aa, device, sample=False):
    vae.eval()
    seqs = {}
    for node, z in latent_vectors.items():
        z_tensor = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            log_probs = vae.decoder(z_tensor).squeeze(0)  # Shape: (nl, nc)
            if sample:
                probs = torch.softmax(log_probs, dim=-1)
                # Sample from the probability distributions represented by the rows
                aa_indices = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                aa_indices = torch.argmax(log_probs, dim=-1)
        seqs[node] = ''.join([idx_to_aa[i] for i in aa_indices.cpu().numpy()])
    return seqs

def main():
    parser = argparse.ArgumentParser(description="Simulate MSAs from VAE latent Brownian motion on trees.")
    parser.add_argument("-l", "--length", type=int, default=97, help="Length of the MSA")
    parser.add_argument("-m", "--model_path", type=str, 
                        default="saved_models/potts/1250/COG28-s1.0-pottsPF00565/model_layers500_ld20_wd0.01_epoch50_2025-06-13.pt", 
                        help="Path to trained VAE model")
    parser.add_argument("-s", "--scaling_factor", type=float, default=1.0, help="Branch length scaling factor")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Device for torch model")
    args = parser.parse_args()

    msa_length = args.length
    scaling_factor = args.scaling_factor
    model_path = args.model_path
    device = args.device

    # Get dictionary mapping amino acids to integers and inverse for the given VAE model 
    model_dir_parts = os.path.dirname(model_path).split(os.sep)
    model_name = os.path.basename(model_path)
    is_trans, ld, num_hidden_units, dim_aa_embed, one_hot = parse_model_name(model_name)
    data_path = os.path.join("msas", model_dir_parts[1], "processed", model_dir_parts[2], model_dir_parts[3])  
    aa_to_idx = aa_to_int_from_path(data_path)
    idx_to_aa = invert_dict(aa_to_idx, unknown_symbol='-')

    # Trees that we will simulate over and output directories
    tree_dirs = {
        '1250': 'trees/fast_trees/1250',
        '5000': 'trees/fast_trees/5000'
    }
    output_dirs = {
        '1250': f'msas/vae/raw/1250',
        '5000': f'msas/vae/raw/5000'
    }

    # Load VAE model
    vae = load_model(model_path, nl=msa_length, nc=21,
                    num_hidden_units=num_hidden_units, nlatent=ld,
                    one_hot=one_hot, dim_aa_embed=dim_aa_embed, trans=is_trans)  
    vae.to(device)

    # Simulate evolution on each tree with Brownian motion followed by decoding to sequeces
    for n_seq in tree_dirs:
        output_dir = output_dirs[n_seq]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        tree_files = glob.glob(os.path.join(tree_dirs[n_seq], '*.clean.tree'))
        for tree_file in tree_files:
            print(f"Processing tree: {tree_file}")
            latent_vectors = simulate_brownian_on_tree(tree_file, latent_dim=ld, scale=scaling_factor)

            seqs = decode_latents_to_seqs(latent_vectors, vae, idx_to_aa, device)
            seq_records = [SeqRecord(Seq(sequence), id=seq_id, description="") for seq_id, sequence in seqs.items()]
            fam_name = os.path.basename(tree_file).split('.')[0]
            output_file = f"{output_dir}/{fam_name}-s{scaling_factor}.fa"
            with open(output_file, 'w') as new:
                SeqIO.write(seq_records, new, "fasta")

if __name__ == "__main__":
    main()
