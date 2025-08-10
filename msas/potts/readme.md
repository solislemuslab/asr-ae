To fit a Potts model, we used the Julia version of adabmDCA 2.0. The Julia package can be installed from the Github repository [https://github.com/spqb/adabmDCA.jl](adabmDCA.jl) and the wrapper scripts `execute.jl` and `adabmDCA.sh` by following the instructions in the README of the Github repository. 

With the wrapper scripts installed in the directory `scripts/adabmDCA/`, we can train the DCA model (with a Boltzmann machine) on the pre-processed MSA of PF00565 located at `msas/real/processed/PF00565/seq_msa_char.fasta` using the following command:
```
scripts/adabmDCA/adabmDCA.sh train -d msas/real/processed/PF00565/seq_msa_char.fasta -o msas/potts --nthreads 4 -l pf00565
```
We can compute the Frobenius contact matrix using
```
scripts/adabmDCA/adabmDCA.sh contacts -p msas/potts/pf00565_params.dat -o msas/potts -l pf00565
```
*NB: Contrary to the published documentation he Julia package source code produces a contact matrix that 1. incorporates the gap character and 2. does not use an Average Product Correction (APC). In order to get a contact matrix that will do 1 and 2, we modify the package's source code.

We can sample a Markov chain of sequences from the resulting Potts model with:
``` 
scripts/adabmDCA/adabmDCA.sh sample -p msas/potts/pf00565_params.dat -d msas/real/processed/PF00565/seq_msa_char.fasta -o msas/potts --nthreads 4 -l pf00565 
```
To reformat the parameters file with integer representations of amino acids so that the Julia function `PottsEvolve.read_graph()` will be able to parse it, run
```
python msas/potts/reformat_potts.py msas/potts/pf00565_params.dat msas/potts/pf00565_params_intindex.dat
```

Finally, to produce the MSAs in this folder, we sample the Markov chain of sequences from our fitted Potts model along the various trees in `trees/fast_trees` by running 
```
julia --project=. scripts/simulate_potts_msas.jl
```