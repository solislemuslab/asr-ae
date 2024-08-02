#!/bin/bash

# Directory containing tree files (which are separated into subdirectories based on the number of taxa, either 5k or 1250)
tree_files_dir="./fast_trees"

# Iterate through each .tree file in the directory
for tree_file in "$tree_files_dir"/*/*.tree; do
    echo "Processing $tree_file"
    # replace all negative branch lengths with 0.0 and get rid of internal node labels (clade supports)
    sed -E 's/-[0-9]\.[0-9]*/0\.0/g; s/(\))[0-9]\.[0-9]*(:)/\1\2/g' "$tree_file" > "$tree_file"_cleaned 
    ./gen_seq_lg.sh "$tree_file"_cleaned
done
