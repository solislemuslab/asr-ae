#!/bin/bash

# Global choices
while getopts "l:s:a:f:" opt
do
   case "$opt" in
      l ) parameterL="$OPTARG" ;;
      s ) parameterS="$OPTARG" ;;
      a ) parameterA="$OPTARG" ;;
      f ) tree_file="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# if parameterA=none, set h to empty string "", else set variable het to the string "-a $parameterA"
if [ "$parameterA" = "None" ]; then
    het=""
else
    het="-a $parameterA"
fi

# Wrangle file paths and create output path name
name=$(basename "$tree_file" .tree_cleaned)
fam_name=${name%%.*} # removes "sim.trim"
n_seq=$(basename "$(dirname "$tree_file")")
output_dir="independent_sims/${n_seq}"
# Create output directory if it doesn't exist
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi
output_file="${output_dir}/${fam_name}-l${parameterL}-s${parameterS}-a${parameterA}_msa.dat"

# Read in the exchange parameters and equilibrium frequencies from the output of the Python script
read -r exchanges freqs < <(python get_lg_params.py)
# Replace commas separating the exchange parameters with spaces
# This is required for seq-gen to read it as a command line argument for some reason
exchanges=${exchanges//,/ } 

# # Run seq-gen with specified parameters and input tree file to generate MSA in the output file in nexus format
seq-gen -mGENERAL -z770 $het -l $parameterL -s $parameterS -f $freqs -r $exchanges -wa -on < $tree_file > $output_file