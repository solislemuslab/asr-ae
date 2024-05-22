# Hailey's Scripts README # 

## IQ-Tree Scripts ##

IQ-Tree Scripts for PF00565, PF00067, PF00041 can be found in `~\asr-ae\scripts\IQ-TREE_Ding_Data_April24.R`

  - These scripts were executed using the `tmux` command to run on the WID server
  
  - These trees are being constructed using the embeddings from Evan, not the actual sequences themselves
  
  - Because of no a priori information given regarding the evolutionary model, we use the `-MFP` command to let IQ-Tree find the best-fitting evolutionary model for each of the 3 datasets, respectively using lowest-BIC criterion.
  
  - Due to computational time / complexity, we use the "Ultrafast bootstrap" option with 1000 replicates (`-bb 1000`), which uses a pseudolikelihood approach to optimizing the tree rather than full likelihood (note: this approach has been shown to be less reliable than the regular bootstrap using full likelihood, but is too computationally complex to complete given the data size).  
  
  - As of May 21st, 2024, only the tree for PF00565 was finished running 
  
  - The tree single tree file for P00565 can be found in `~\asr-ae\data\iqtree\tree_files`
  

## ASR for Embeddings 

ASR Script for PF00565 can be found in `~\asr-ae\scripts\embeddings_asr.Rmd`

  - We decided to use `MVMorph` package to model the multivariate ancestral sequence evolution trajectory
  
  - Hailey tried performing the ASR during the week of 5/13 - 5/17 using this package, but both her local version of R and the WID server's crashed (likely due to computational load)
  
  - Hailey worked with Em and Emile in IT to try and diagnose the problem.  

  - Em Craft ran the script line-by-line in WID server and had this to say: 
  
    "I commented out each line of your test script until I could see where the failure happens. It’s only     on the last line:"
                
    fit.pf00565 <- mvgls(pf00565.matrix~1, data=pf00565.list, pf00565.tree, model="BM", method="LL", REML = FALSE)
    
  - Em also received this message from the `mvgls()` function when it terminated: "There are more variables than observations. Please try instead the penalized methods "RidgeArch", "RidgeAlt" or "LASSO"
  
  





