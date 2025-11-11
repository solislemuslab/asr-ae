# Reproducing our analysis


## Pipeline for benchmarking on simulated data

This is the pipeline for how we benchmarked the VAE-based approach to ancestral sequence reconstruction on simulated data.

### Simulating data

1. First, we simulate an MSA. 

To simulate sequences to evolve with independent sites according to the LG model, run
```
bash scripts/simulate_lg_msas.sh [-l length] [-s scale] [-a gamma_rate] [-t tree]
```
This will generate an MSA file in `msas/independent/raw`

Conretely, for the results with independent sites in the paper, we simulated evolution along the cleaned COG28 tree (which has 1246 tips) and the cleaned COG2814 tree (which has 4952 tips) with a gamma rate heterogeneity parameters of 0.5 using the commands 
```
bash scripts/simulate_lg_msas.sh -a 0.5 -t trees/fast_trees/1250/COG28.clean.tree 
``` 
and 
```
bash scripts/simulate_lg_msas.sh -a 0.5 -t trees/fast_trees/5000/COG2814.clean.tree 
``` 
We simulated evolution with inalong the 

To evolve an MSA with epistasis or co-evolution requires an existing MSA to which to fit a generative model. For this purpose, we use a pre-processed MSA of Staphylococcal nuclease homologs, located at `msas/real/processed/PF00565/seq_msa_char.fasta`. 

With this MSA existing at this path, we can simulate sequences to evolve according to the autoregressive epistatic model fit to this MSA by running
```
julia scripts/simulate_ardca_msas.jl
```
This will generate an MSA file in `msas/ardca/raw`.

To evolve an MSA with the Potts epistatic model, we first need to fit a Potts model and save the parameters to disk. We can do this by following the steps in msas/



2. Next, we first have to process the MSA to generate the Python objects we'll need for our Variational AutoEncoder. To do this, we run 

```
python msas/scripts/process_msa.py msas/independent/raw/1250/COG28-l60-s1-a0.5.fa
```

This command will generate a directory in `msas/independent/processed/1250` called `COG28-l100-s1-a0.5`. In this directory, you will find all the Python objects (as `.pkl` files) that represent the processed MSA for the purposes of the VAE, as well as a fasta file of the processed MSA. As opposed to the raw MSA generated in step 1, this processed MSA does not include the ancestral sequences.

3. The next step is to train the VAE! Check out the file `autoencoder/config.json`. This file contains configuration details for the training. You can manually edit the details of this file. When you have all the configs you want, simply run from the project root directory 

```
python autoencoder/train.py autoencoder/config.json
```
The value of test and training loss, as well as test reconstruction accuracy, after every epoch will get printed to standard output. If `plot_results` is true, a plot of the learning curve will be saved in `plots/independent/1250/COG28-l100-s1-aNone` after training completes. If `save_model` is true, the model will get saved in `saved_models/independent/1250/COG28-l100-s1-aNone`

4. The next step is to generate embeddings of the leaf sequences from this model. The file `embeddings/config.json` specifies the dataset (MSA) and fitted model you want to generate embeddings for/from. Simply run 

```
python embeddings/gen_embeddings.py embeddings/config_gen.json
```
This will write the embeddings as a CSV file to the folder `embeddings/data/independent/1250/COG28-l100-s1-aNone`. It also plots the embeddings (or a reduced dimensional version of them with PCA) in 2d space.

5. Now that we have embedding of the leaf sequences, we want to use a Brownian motion model to infer embeddings at the internal nodes. In other words, we want to reconstruct "ancestral embeddings". For this purpose, we use the R package **Rphylopars**, which implements scalable multivariate phylogenetic comparative analysis. To obtain ancestral reconstruction of embeddings, run 

```
Rscript embeddings/embeddings_asr.R msas/independent/processed/1250/COG28-l100-s1-aNone name_of_model
```

The script will write a csv of ancestral embeddings into the same folder as the embeddings of the sequences at the tips from step 4, and it will also produce a plot that shows the phylogenetic tree ploted as a network with the coordinates of the nodes given by the (first two dimensions of the) embeddings (for tree tips, coordinates are directly from the VAE encoder and for the internal nodes, coordinates are the estimated "reconstructions" from the Brownian motion model).

Because the ancestral reconstruction algorithm cannot handle extremely short external branches (< .001), we trim the tree of these branches before reconstructing the embeddings. The modified tree is saved to the tree directory with suffix `fully_trim.tree` instead of just `trim.tree`. The number of remaining tips and internal nodes in the trimmed tree is printed, and the names of the remaining tips are saved to the file `final_seq_names.txt` in the `processed` directory.


6. Last but not least, let's decode the reconstructed embeddings back to reconstructed sequences. Running 
```
python embeddings/decode_recon_embeds.py embeddings/config_decode.json 
```
will do this and evaluate the reconstructed ancestral sequences against the true ancestral sequences, calculating the identity percentage metric, and comparing it to a baseline of predicting the modal amino acid at each position for each ancestor and a baseline of IQTree using a correctly specified LG model.