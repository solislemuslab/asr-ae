#!/bin/bash
# Read in the exchange parameters and equilibrium frequencies from the output of the Python script
read -r exchanges freqs < <(python get_aa_params.py)
echo -e "Exchange parameters:\n$exchanges"
echo -e "Equilibrium frequencies:\n$freqs"
# Replace commas separating the exchange parameters with spaces
# For some reason, this is required for seq-gen to read it as a command line argument 
exchanges=${exchanges//,/ } 
# Run seq-gen with the specified parameters and the tree file
seq-gen  -mGENERAL -z770 -l100 -wa -on -f $freqs -r $exchanges < ../iqtree/tree_files/pf00565_rerooted.tree > pf00565_simulated.dat