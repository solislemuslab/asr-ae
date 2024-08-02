#!/bin/bash

# Check that a tree file was provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <tree_file>"
    exit 1
fi

# Global choices
l=150 # sequence length
s=0.5 # branch length scaling factor

# Wrangle file paths and create output path name
name=$(basename "$1" .tree_cleaned)
fam_name=${name%%.*} # removes "sim.trim"
n_seq=$(basename "$(dirname "$1")")
output_dir="msas/${n_seq}"
# Create output directory if it doesn't exist
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi
output_file="${output_dir}/${fam_name}-l${l}-s${s}_msa.dat"

# Read in the exchange parameters and equilibrium frequencies from the output of the Python script
read -r exchanges freqs < <(python get_lg_params.py)
# Replace commas separating the exchange parameters with spaces
# This is required for seq-gen to read it as a command line argument for some reason
exchanges=${exchanges//,/ } 
# Run seq-gen with specified parameters and input tree file to generate MSA in the output file in nexus format
seq-gen  -mGENERAL -z770 -l $l -s $s -f $freqs -r $exchanges -wa -on < "$1" > "$output_file"