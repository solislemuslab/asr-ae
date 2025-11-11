## Environment set-up
This repository includes Python, R, and Julia code.


### Julia

From the project directory, start a REPL with ```julia --project=.``` Enter the Pkg REPL with `]` and run `instantiate`.

### Python
With conda installed, run 

```
conda env create -f environment.yml
```
This will create a new conda environment called `torch`. Activate this environment with
```
conda activate torch
```

### R
For R, the main packages used are those in the `tidyverse`, `ape`, `Rphylopars`, and `network`. 


## Pipeline for benchmarking on simulated data

This is the pipeline for how we benchmarked the VAE-based approach to ancestral sequence reconstruction on simulated data.

### Simulating data

1. First, we simulate an MSA. 

To simulate sequences to evolve with independent sites according to the LG model, run
```
bash scripts/simulate_lg_msas.sh [-l length] [-s scale] [-a gamma_rate] [-t tree]
```
This will generate an MSA file in `msas/independent/raw`

Concretely, for the results with independent sites in the paper, we simulated evolution along the cleaned COG28 tree (which has 1246 tips) and the cleaned COG2814 tree (which has 4952 tips) with a gamma rate heterogeneity parameters of 0.5 using the commands 
```
bash scripts/simulate_lg_msas.sh -a 0.5 -t trees/fast_trees/1250/COG28.clean.tree 
``` 
and 
```
bash scripts/simulate_lg_msas.sh -a 0.5 -t trees/fast_trees/5000/COG2814.clean.tree 
``` 
We simulated independent evolution on the 10,000-tip tree with no rate heterogeneity with:
```
bash scripts/simulate_lg_msas.sh -t trees/10000/pevae.clean.tree 
```

To evolve an MSA with epistasis or co-evolution requires an existing MSA to which to fit a generative model. For this purpose, we use a pre-processed MSA of Staphylococcal nuclease homologs, located at `msas/real/processed/PF00565/seq_msa_char.fasta`. 

With this MSA existing at this path, we can simulate sequences to evolve according to the autoregressive epistatic model fit to this MSA by running
```
julia --project=. scripts/simulate_ardca_msas.jl
```
This will generate MSA files in `msas/ardca/raw`.

To evolve an MSA with the Potts epistatic model, we first need to fit a Potts model and save the parameters to disk. We can do this by following the steps in `msas/potts/readme.md`. Once we have a saved a Potts model parameters file saved at `msas/potts/pf00565_params_intindex.dat`, we can evolve the MSA according to the Potts model by running 

```
python --project=. scripts/simulate_potts_msas.jl
```
This will generate MSA files in `msas/potts/raw`


2. Next, we pre-process the MSA and generate some pickled Python objects that will be used for training the VAE. To do this, we run 
```
python scripts/process_msa.py <MSA-file>
```
For example, if we wanted to continue the benchmarking pipeline with one of the MSAs we evolved under the independent LG model in the previous step, we could run
```
python scripts/process_msa.py msas/independent/raw/1250/COG28-l100-s1-a0.5.fa
```
This command will generate a directory in `msas/independent/processed/1250` called `COG28-l100-s1-a0.5`. In this directory, you will find all the Python objects (as `.pkl` files) that represent the processed MSA for the purposes of the VAE, as well as a fasta file of the processed MSA. As opposed to the raw MSA generated in step 1, this processed MSA does not include the ancestral sequences.


3. The next step is to train the VAE! Check out the file `config.json`. This file contains configuration details for the training. You can manually edit the details of this file. When you have specified all the configs you want, including the path to the folder with the saved data files that we just generated, `msas/independent/processed/1250/COG28-l100-s1-a0.5`,  simply run
```
python autoencoder/train.py config.json
```
The value of test and training loss, as well as test reconstruction accuracy, after every epoch will get printed to standard output. If `plot_results` is true, a plot of the learning curve will be saved in `plots/independent/1250/COG28-l100-s1-aNone` after training completes. If `save_model` is true, the model will get saved in `saved_models/independent/1250/COG28-l100-s1-aNone`

4. The next step is to generate embeddings of all the sequences from this model (including the ancestral ones). Again, the file `config.json` has the relevant configs (under the heading `generate`) that you can manually edit. Make sure that `model_name` is the name of the model you just trained. Then simply run 
```
python embeddings/gen_embeddings.py config.json
```
This will write the embeddings as a CSV file to the folder `embeddings/data/independent/1250/COG28-l100-s1-aNone`. It also plots `plots/independent/1250/COG28-l100-s1-aNone` the embeddings (or a reduced dimensional version of them with PCA) in 2d space.

5. Now that we have embedding of the leaf sequences, we want to use a Brownian motion model to infer embeddings at the internal nodes. Note that in the previous step, we generated embeddings for the ancestral sequences as well because we know them in this simulated setting, but in practice, we would only have embeddings of the leaf sequences and we'd have to use these to estimate the embeddings of internal nodes. In other words, we want to reconstruct "ancestral embeddings". For this purpose, we use the R package **Rphylopars**, which implements scalable multivariate phylogenetic comparative analysis and in particular allows us to fit multivariate phylogenetic Brownian motion. To obtain ancestral reconstruction of embeddings, run 

```
Rscript embeddings/embeddings_asr.R msas/independent/processed/1250/COG28-l100-s1-aNone name_of_model
```

The script will write a csv of ancestral embeddings into the same folder as the embeddings of the sequences at the tips from step 4, and it will also produce a plot that shows the phylogenetic tree ploted as a network with the coordinates of the nodes given by the (first two dimensions of the) embeddings (for tree tips, coordinates are directly from the VAE encoder and for the internal nodes, coordinates are the estimated "reconstructions" from the Brownian motion model). 


If we simulated the evolution, which we did in this case, it will also produces a plot that shows arrows connecting the estimated ancestral embeddings to the true ancestral embeddings that we got in step 4.


6. Last but not least, let's decode the reconstructed embeddings back to reconstructed sequences. Running 
```
python embeddings/decode_recon_embeds.py embeddings/config_decode.json 
```
will do this and evaluate the reconstructed ancestral sequences against the true ancestral sequences, calculating the accuracy for each ancestral sequence and printing average Hamming accuracy over all ancestral sequences to standard output. It will do this also for a few other baseline methods of ancestral sequence reconstruction (including the extreme baseline of just predicting the consensus amino acid from the MSA at each position for each ancestor). It will also produce a plot with a LOESS smooth of the scatter plot of the ancestor's depth in the tree (distance to nearest leaf) versus the Hamming distance of the sequence reconstruction for each of the methods.


*NB: `run_pipeline.sh` is a wrapper to run all of steps 3-6 below. Just set the specifications in `config.json` and run `run_pipeline.sh config.json` and it will do all four steps (training the VAE, generating embeddings, reconstructing ancestral embeddings, and decoding back to sequences and evaluating)*


## Real protein families
See `msas/real/readme.md`.


## Figures in paper
The Jupyter notebooks in `notebooks` produce most of the figures from the paper. 

- `notebooks/figure3.ipynb` for Figure 3
- `notebooks/figures4-5.ipynb` for Figures 4 and 5
- `reconstruct_anc_pf00144.ipynb` for Figure 6 and B3
- `analyze_model_real.ipynb` for Figure B2
- `epistasis_checks.ipynb` for Figure A1

Note that these notebooks depend on models and embeddings having been saved at the appropriate path using the pipeline for running the method described above.