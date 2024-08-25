#!/bin/bash
helpFunction()
{
   echo ""
   echo "Usage: $0 [-l parameterL] [-s parameterS] [-h parameterH]"
   echo -e "\t-l Sequence length (defualt 150)"
   echo -e "\t-s Branch length scaling factor (default 1)"
   echo -e "\t-a Gamma rate heterogeneity (either positive real number or \"None\")"
   exit 1 # Exit script after printing help
}

# Get options
while getopts "l:s:a:" opt
do
   case "$opt" in
      l ) parameterL="$OPTARG" ;;
      s ) parameterS="$OPTARG" ;;
      a ) parameterA="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Set non-provided options to default values
if [ -z "$parameterL" ] 
then
   echo "Default branch length of 100 will be used";
   parameterL=100
fi

if [ -z "$parameterS" ] 
then
   echo "Default branch length scaling factor of 1 will be used";
   parameterS=1
fi

if [ -z "$parameterA" ] 
then
   echo "No rate heterogeneity will be used";
   parameterA="None"
fi

# Directory containing tree files (which are separated into subdirectories based on the number of taxa, either 5k or 1250)
tree_files_dir="./fast_trees"

# Iterate through each .tree file in the directory
for tree_file in "$tree_files_dir"/*/COG*.tree; do
    echo "Processing $tree_file"
    # replace all negative branch lengths with 0.0 and get rid of internal node labels (clade supports)
    sed -E 's/-[0-9]\.[0-9]*/0\.0/g; s/(\))[0-9]\.[0-9]*(:)/\1\2/g' "$tree_file" > "$tree_file"_cleaned
    # run gen_seq_lg.sh with the cleaned tree file 
    ./gen_seq_lg.sh -l $parameterL -s $parameterS -a $parameterA -f "$tree_file"_cleaned
done
