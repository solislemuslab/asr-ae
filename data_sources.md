# Not using so far
- PSICOV MSAs for a bunch of PFAM families. Currently have saved in `ccmgen-scripts` folder
- Phylogeny-MSATransformer/bmDCA MSAs for a bunch of PFAM families (can be found in [GitHub repo](https://github.com/Bitbol-Lab/Phylogeny-MSA-Transformer/tree/main/data/Pfam_full/msa), and description of how data is obtained in Nature article section "Datasets")

# Using so far
## Real
- Sequences from *only Eukaryotic species* of three PFAM families investigated by Ding: PF00565, PF00041, and PF00067
## Simulated
- Independent evolution
    - Backbone trees on which we evolved sequences are taken from the FastTree study (downloaded from [here](http://www.microbesonline.org/fasttree/)). Trees were inferred to a collection of sequences from a Cluster of Orthologous Groups gene family (aligned to the family's profile)
        - First group of trees were inferred with PhyML to an allignment of 1250 sequences 
        - Second group of trees were inferred with FastTree to an allignment of 5000 sequences 
    - We then used SeqGen to simulate sequences on these backbone trees using the LG model with evolution happening at equal rate and independently across sites.
- Evolution with coupling
    - Using CCMGen or using Bitbol-Lab/Phylogeny-Partners implementation, evolve sequences along the same trees used for the independent evolution simulations

