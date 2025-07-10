#!/bin/bash

helpFunction() {
    echo ""
    echo "Usage: $0 [-l length] [-s scale] [-a gamma_rate] [-t tree_file]"
    echo -e "\t-l Sequence length (default 100)"
    echo -e "\t-s Branch length scaling factor (default 1)"
    echo -e "\t-a Gamma rate heterogeneity (positive real number or \"None\")"
    echo -e "\t-t Specific tree file to simulate on (optional)"
    exit 1 # Exit script after printing help
}

# Get options
while getopts "l:s:a:t:" opt; do
    case "$opt" in
        l ) seq_length="$OPTARG" ;;
        s ) scale="$OPTARG" ;;
        a ) het="$OPTARG" ;;
        t ) single_tree="$OPTARG" ;;
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
    esac
done

# Set non-provided options to default values
if [ -z "$seq_length" ]; then
    echo "Default sequence length of 100 will be used"
    seq_length=100
fi

if [ -z "$scale" ]; then
    echo "Default branch length scaling factor of 1 will be used"
    scale=1
fi

if [ -z "$het" ]; then
    echo "No rate heterogeneity will be used"
    het="None"
fi

if [ "$het" = "None" ]; then
    include_het=""
else
    include_het="+G{$het}" #discrete gamma rate heterogeneity with 4 categories and shape parameter $het
fi

if [ "$scale" = "1" ]; then
    include_scale=""
else
    include_scale="--branch-scale $scale"
fi 

param_string="l${seq_length}-s${scale}-a${het}"

# Directory containing tree files (which are separated into subdirectories based on the number of taxa, either 5000 or 1250)
tree_files_dir="trees/fast_trees"

seed=770 # random seed for simulating MSA

if [ -n "$single_tree" ]; then
    # User specified a single tree file
    if [ ! -f "$single_tree" ]; then
        echo "Specified tree file does not exist: $single_tree"
        exit 1
    fi
    # Infer n_seq from the path (assumes path contains /5000/ or /1250/)
    if [[ "$single_tree" =~ /5000/ ]]; then
        n_seq="5000"
    elif [[ "$single_tree" =~ /1250/ ]]; then
        n_seq="1250"
    else
        echo "Could not determine n_seq (5000 or 1250) from tree file path."
        exit 1
    fi
    output_dir="msas/independent/raw/$n_seq"
    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
    fi
    fam_name=$(basename "$single_tree" .clean.tree)
    output_file="${output_dir}/${fam_name}-${param_string}"
    iqtree_command="iqtree/bin/iqtree2 --alisim $output_file \
                    -seed $seed \
                    -m LG$include_het \
                    --length $seq_length \
                    $include_scale \
                    --write-all \
                    -nt AUTO \
                    --out-format fasta \
                    -quiet \
                    -t $single_tree"
    $iqtree_command
    rm "$single_tree.log"
else
    # Iterate through each .tree file in the directory
    for n_seq in "5000" "1250"; do
        output_dir="msas/independent/raw/$n_seq"
        if [ ! -d "$output_dir" ]; then
            mkdir -p "$output_dir"
        fi
        for tree_file in $tree_files_dir/$n_seq/COG*.clean.tree; do
            fam_name=$(basename "$tree_file" .clean.tree)
            output_file="${output_dir}/${fam_name}-${param_string}"
            iqtree_command="iqtree/bin/iqtree2 --alisim $output_file \
                            -seed $seed \
                            -m LG$include_het \
                            --length $seq_length \
                            $include_scale \
                            --write-all \
                            -nt AUTO \
                            --out-format fasta \
                            -quiet \
                            -t $tree_file"
            $iqtree_command
            rm "$tree_file.log"
            seed=$((seed+1))
        done
    done
fi
