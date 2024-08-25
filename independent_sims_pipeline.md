1. We start with the trees found at `data/simulations/fast_trees/*/COG*.sim.trim.tree`. As discussed in `data/data_sources.md`, this tree is taken from the FastTree study. The first step in the pipeline is to run 

```
cd data/simulations
./gen_all_msas.sh [-l sequence length] [-s branch length scaling factor] [-a Gamma rate heterogeneity]
``` 

This will generate one simulated MSA per tree in subdirectores `5k` and `1250` of `data/simulations/msas`. The defaults for the three optional arguments are 100, 1, and "none", respectively. The sites all evolve independently.

2. Next, choose an MSA to work with. Let's say I want to work with the COG28 family with 1250 sequences. We have to process the MSA to generate the Python objects we'll need for our Variational AutoEncoder. To do this, we run 