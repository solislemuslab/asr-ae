To train a Potts model, we used the Julia implementation of adabmDCA 2.0, i.e. the Julia package [`adabmDCA.jl`](https://github.com/evangorstein/adabmDCA.jl), which is contained in our Julia environment. The wrapper scripts to run the functionality from the package have also been downloaded in `scripts/adabmDCA` (and slightly edited).

We fit a DCA model to the pre-processed MSA of PF00565 with
```
scripts/adabmDCA/adabmDCA.sh train -d msas/real/processed/PF00565/seq_msa_char.fasta -o msas/potts --nthreads 4 --target 0.90 -l pf00565
```
This program uses multi-threading in Julia (with the number of threads specified with the `--nthreads` flag). Training stops when the Pearson correlation on the two-point statistics reaches the `--target` value (here 0.90). All output is redirected to a log file (`<label>_adabmDCA.log` in the output directory).

As an alternative, you can use the Python version of adabmDCA 2.0. A Colab notebook that uses the Python version and allows uploading the MSA will produce the same result as the Julia implementation we used: https://colab.research.google.com/drive/1l5e1W8pk4cB92JAlBElLzpkEk6Hdjk7B?usp=sharing#scrollTo=t5nf2gHcIBmd

Training can be resumed from the most recent saved parameters and Markov chains (saved every 50 epochs) by running
```
scripts/adabmDCA/adabmDCA.sh train -d msas/real/processed/PF00565/seq_msa_char.fasta -o msas/potts --nthreads 4 -l pf00565 -p msas/potts/pf00565_params.dat -c msas/potts/pf00565_chains.fasta
```

We can compute the contact matrix from the fitted parameters using
```
scripts/adabmDCA/adabmDCA.sh contacts -p msas/potts/pf00565_params.dat -o msas/potts -l pf00565
```

To reformat the parameters file with integer representations of amino acids so that the Julia function `PottsEvolve.read_graph()` will be able to parse it, run
```
python msas/potts/reformat_potts.py msas/potts/pf00565_params.dat msas/potts/pf00565_params_intindex.dat
```

Finally, to produce the MSAs in this folder (`msas/potts/raw/`), we sample the Markov chain of sequences from our fitted Potts model along the various trees in `trees/fast_trees` by running 
```
julia --project=. scripts/simulate_potts_msas.jl
```