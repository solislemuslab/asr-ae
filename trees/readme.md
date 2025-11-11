These trees are available from http://www.microbesonline.org/fasttree/downloads/aa5K_new.tar.gz (trees with 5k leaves) and http://www.microbesonline.org/fasttree/downloads/aa1250.tar.gz (trees with 1250 leaves). The original Fasttree paper explains how these trees were estimated based on alignments of protein families from the Clusters of Orthologous Groups (COG), using either PhyML (for the 1250 leaf trees) or FastTree (for the 5k leaf trees). Note that the FastTree algorithm included some negative external branch lengths in the 5k leaf trees. 

We clean these trees with...

We cluster these trees using TreeCluster (Balaban et al) with their Max method. For example, for `COG277.clean.tree` we run:
```
 TreeCluster.py -i trees/fast_trees/1250/COG277.clean.tree -o trees/fast_trees/1250/COG277_clusters.txt -m MAX -t 3
```


## Inferred trees

The trees in `trees/inferred_from_real` were inferred from real protein families using IQTree. In particular, the phylogenetic tree for PF00144 saved at `trees/inferred_from_real/PF00144.treefile` was inferred from the pre-processed MSAs at `msas/real/processed/PF00144_og/seq_msa_char.fasta` using the following command:

```
iqtree/bin/iqtree2 -s msas/real/processed/PF00144_og/seq_msa_char.fasta -m MFP -bb 1000
```

Because no prior information was available regarding the appropriate evolutionary model, we enabled automatic model selection using the `-m MFP` option, which evaluates a range of candidate substitution models and selects the best-fitting model based on the lowest Bayesian Information Criterion (BIC).