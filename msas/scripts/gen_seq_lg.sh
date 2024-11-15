#!/bin/bash

# Global choices
while getopts "z:l:s:a:f:" opt
do
   case "$opt" in
      z ) seed="$OPTARG" ;;
      l ) seq_length="$OPTARG" ;;
      s ) scale="$OPTARG" ;;
      a ) het="$OPTARG" ;;
      f ) tree_file="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# if het=none, set het to empty string "", else set variable het to the string "-a $parameterA"
if [ "$het" = "None" ]; then
    include_het=""
else
    include_het="-a $het"
fi

# Wrangle file paths and create output path name
name=$(basename "$tree_file" .tree_cleaned)
fam_name=${name%%.*} # removes "sim.trim"
n_seq=$(basename "$(dirname "$tree_file")")
output_dir="independent_sims/raw/${n_seq}"
# Create output directory if it doesn't exist
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi
output_file="${output_dir}/${fam_name}-l${seq_length}-s${scale}-a${het}_msa.dat"

# Read in the exchange parameters and equilibrium frequencies from the output of the Python script
read -r exchanges freqs < <(python scripts/get_lg_params.py)
# Replace commas separating the exchange parameters with spaces, required for seq-gen to read it as a command line argument for some reason
exchanges=${exchanges//,/ } 
# Run seq-gen with specified parameters and input tree file to generate MSA in the output file
seq-gen -mGENERAL $include_het -z$seed -l$seq_length -s$scale -f$freqs -r$exchanges -wa -q < $tree_file > $output_file