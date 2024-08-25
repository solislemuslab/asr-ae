from seq_io_utils import to_fasta
import sys
import os

# Path to IQ-TREE executable
iqtree = "data/real/iqtree/bin/iqtree2"
# receive path to MSA as command line argument
msa_path = sys.argv[1]
msa_dir = "/".join(msa_path.split("/")[:-1])
# receive model to use for iqtree reconstruction as command line argument
if len(sys.argv) > 2:
    model = sys.argv[2]
else:
    model = "LG"
# Get sequence names to keep
with open(f"{msa_dir}/final_seq_names.txt") as names:
    keep = names.read().splitlines()
# convert MSA to FASTA format, keeping only the sequences in keep
msa_path_fasta = msa_path.replace(".txt", ".fasta")
to_fasta(msa_path, msa_path_fasta, keep)
# Tree path
family = msa_path.split("/")[-2].split("-")[0]
tree_dir = "data/simulations/fast_trees"
if os.path.exists(f"{tree_dir}/1250/{family}.sim.trim.tree_processed"):
    tree_file = f"{tree_dir}/1250/{family}.sim.trim.tree_processed"
elif os.path.exists(f"{tree_dir}/5k/{family}.sim.trim.tree_processed"):
    tree_file = f"{tree_dir}/5k/{family}.sim.trim.tree_processed"
else:
    print(f"Processed tree file not found for family {family}")
    sys.exit(1)
# Run IQTree command
os.system(f"{iqtree} -redo -s {msa_path_fasta} -m {model} -te {tree_file} -asr")


