# Hailey's Scripts README # 

## IQ-Tree Scripts ##

IQ-Tree Scripts for PF00565, PF00067, PF00041 can be found in `iqtree\IQ-TREE_Ding_Data_April24.R`

  - These scripts were executed using the `tmux` command to run on the WID server
  
  - These trees are being constructed based on the processed MSAs from Evan, located at `msas/real/processed/pfam_id/seq_msa_char_pf*.fasta`. This is what is fed to IQTree
  
  - Because of no a priori information given regarding the evolutionary model, we use the `-MFP` command to let IQ-Tree find the best-fitting evolutionary model for each of the 3 datasets, respectively using lowest-BIC criterion.
  
  - Due to computational time / complexity, we use the "Ultrafast bootstrap" option with 1000 replicates (`-bb 1000`), which uses a pseudolikelihood approach to optimizing the tree rather than full likelihood (note: this approach has been shown to be less reliable than the regular bootstrap using full likelihood, but is too computationally complex to complete given the data size).  
  
  - As of May 21st, 2024, only the tree for PF00565 was finished running 
  
  - The tree single tree file for P00565 can be found in `trees\inferred_real_trees\pf00565`
  

