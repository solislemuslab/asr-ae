A phylogenetic tree for PF00144 was inferred from the pre-processed MSAs at `msas/real/processed/PF00144_og/seq_msa_char.fasta` using IQTree 2 \citep{Minh2020}. Because no prior information was available regarding the appropriate evolutionary model, we enabled automatic model selection using the `-m MFP` option, which evaluates a range of candidate substitution models and selects the best-fitting model based on the lowest Bayesian Information Criterion (BIC).

In particular, we run
```
iqtree/bin/iqtree2 -s msas/real/processed/PF00144_og/seq_msa_char.fasta -m MFP -bb 1000
```