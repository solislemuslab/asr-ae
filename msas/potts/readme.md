To train DCA model on the processed PF00565 MSA, run from the root directory 
```
scripts/adabmDCA/adabmDCA.sh train -d msas/real/processed/PF00565/seq_msa_char.fasta -o msas/potts --nthreads 8 -l pf00565
```

To sample from this model, run
``` 
scripts/adabmDCA/adabmDCA.sh sample -p msas/potts/pf000565_params.dat -d msas/real/processed/PF00565/seq_msa_char.fasta -o msas/potts -l pf00565 --nthreads 8
```

To reformat the parameters file with integer representations of amino acids so that `PottsEvolve.read_graph()` can parse it, run
```
python msas/potts/reformat_potts.py msas/potts/pf00565_params.dat 
```
