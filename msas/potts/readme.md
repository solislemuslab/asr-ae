To fit a Potts model, we used the Julia version of adabmDCA 2.0. The Julia package can be installed from the Github repository [https://github.com/spqb/adabmDCA.jl](adabmDCA.jl) and the wrapper scripts `execute.jl` and `adabmDCA.sh` by following the instructions in the README of the Github repository. 

With the wrapper scripts installed in the directory `scripts/adabmDCA/`, we can train the DCA model on the pre-processed MSA of PF00565 located at `msas/real/processed/PF00565/seq_msa_char.fasta` using the following command:
```
scripts/adabmDCA/adabmDCA.sh train -d msas/real/processed/PF00565/seq_msa_char.fasta -o msas/potts --nthreads 8 -l pf00565
```

We can sample a Markov chain of sequences from the resulting Potts model with:
``` 
scripts/adabmDCA/adabmDCA.sh sample -p msas/potts/pf000565_params.dat -d msas/real/processed/PF00565/seq_msa_char.fasta -o msas/potts --nthreads 8 -l pf00565 
```

To reformat the parameters file with integer representations of amino acids so that `PottsEvolve.read_graph()` will be able to parse it, run
```
python msas/potts/reformat_potts.py msas/potts/pf00565_params.dat 
```

Finally, to produce the MSAs in this folder, we sample the Markov chain of sequences from our fitted Potts model along the various trees in `trees/fast_trees` by running 

```
julia --project=. scripts/simulate_potts_msas.jl
```