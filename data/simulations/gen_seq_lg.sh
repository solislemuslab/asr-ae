#!/bin/bash

# Check that a tree file was provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <tree_file>"
    exit 1
fi
# Wrangle file paths and create output path name
fam_name=$(basename "$1" .tree_cleaned)
n_seq=$(basename "$(dirname "$1")")
output_dir="msas/${n_seq}"
output_file="${output_dir}/${fam_name}_msa.dat"
# Create output directory if it doesn't exist
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

# Read in the exchange parameters and equilibrium frequencies from the output of the Python script
read -r exchanges freqs < <(python get_lg_params.py)
# Replace commas separating the exchange parameters with spaces, as this is required for seq-gen to read it as a command line argument for some reason
exchanges=${exchanges//,/ } 
# Run seq-gen with the specified parameters and the tree file, including the ancestral sequences
seq-gen  -mGENERAL -z770 -l100 -wa -on -f $freqs -r $exchanges  < "$1" > "$output_file"