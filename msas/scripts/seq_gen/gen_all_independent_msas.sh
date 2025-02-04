#!/bin/bash
helpFunction()
{
   echo ""
   echo "Usage: $0 [-l parameterL] [-s parameterS] [-h parameterH]"
   echo -e "\t-l Sequence length (default 100)"
   echo -e "\t-s Branch length scaling factor (default 1)"
   echo -e "\t-a Gamma rate heterogeneity (either positive real number or \"None\")"
   exit 1 # Exit script after printing help
}

# Get options
while getopts "l:s:a:" opt
do
   case "$opt" in
      l ) seq_length="$OPTARG" ;;
      s ) scale="$OPTARG" ;;
      a ) het="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Set non-provided options to default values
if [ -z "$seq_length" ] 
then
   echo "Default sequence length of 100 will be used";
   seq_length=100
fi

if [ -z "$scale" ] 
then
   echo "Default branch length scaling factor of 1 will be used";
   scale=1
fi

if [ -z "$het" ] 
then
   echo "No rate heterogeneity will be used";
   het="None"
fi

# Directory containing tree files (which are separated into subdirectories based on the number of taxa, either 5k or 1250)
tree_files_dir="../trees/fast_trees"
# Iterate through each .tree file in the directory
seed=770 # random seed for simulating MSA
for tree_file in "$tree_files_dir"/*/COG*.tree; do
   echo "Evolving sequences along $tree_file"
   # replace all negative branch lengths with 0.0 and get rid of internal node labels (clade supports)
   sed -E 's/-[0-9]\.[0-9]*/0\.0/g; s/(\))[0-9]\.[0-9]*(:)/\1\2/g' "$tree_file" > "$tree_file"_cleaned
   # run gen_seq_lg.sh with the cleaned tree file 
   ./scripts/gen_seq_lg.sh -z $seed -l $seq_length -s $scale -a $het -f "$tree_file"_cleaned
   # increment the random seed for simulating MSA for the next tree
   seed=$((seed+1))
done
