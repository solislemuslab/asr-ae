This is the entire pipeline for executing and benchmarking our method on an MSA simulated with independent sites.

1. We start with the trees found at `trees/fast_trees/*/COG*.sim.trim.tree`. As discussed in `data_sources.md`, this tree is taken from the FastTree study. The first step in the pipeline is to run 

```
cd msas
./scripts/gen_all_independent_msas.sh [-l sequence length] [-s branch length scaling factor] [-a Gamma rate heterogeneity]
``` 

The defaults for the three optional arguments are 100, 1, and "none", respectively. This command will generate one simulated MSA per tree in subdirectores `5k` and `1250` of `msas/independent_sims/raw`, in which sites have evolved independently. Before simulating an MSA on a tree, this script creates a new tree file in the tree diretory (with "_cleaned" appended to the original file name) in the tree directory that has replaced any negative branch lengths in the tree with a branch of length 0.0. *NOTE: These MSA files include the ancestral sequences, i.e. the sequences at the internal nodes of the tree. These internal node sequences are labelled with numbers that are larger than the number of tips, i.e. > 1250 if there are 1250 tips. In contrast, the sequences at the tips have labels that begin with the letter N.*

2. Next, choose an MSA to work with. Let's say I want to work with the COG28 family with 1250 sequences. We have to process the MSA to generate the Python objects we'll need for our Variational AutoEncoder. To do this, we run 

```
python scripts/process_msa.py independent_sims/raw/1250/COG28-l100-s1-aNone_msa.dat N1 
```
The "N1" argument tells the script we want to specify sequence N1 as a reference sequence. The query sequences are more relevant for when we're processing the MSAs from real PFAM family. For the simulated MSAs, we arbitrarilly choose the first sequence N1 as our reference.

This command will generate a directory in `msas/independent_sims/processed/1250` called `COG28-l100-s1-aNone`. In this directory, you will find all the Python objects (as `.pkl` files) that represent the processed MSA for the purposes of the VAE, as well as a text file version of the processed MSA. As opposed to the raw MSA generated in step 1, this processed MSA does not include the ancestral sequences. The processed MSA also has any exact duplicates of leaf sequences removed.

3. The next step is to train the VAE! Check out the file `autoencoder/config.json`. This file contains configuration details for the training. You can manually edit the details of this file. When you have all the configs you want, simply run from the project root directory 

```
python autoencoder/train.py autoencoder/config.json
```
The value of test and training loss, as well as test reconstruction accuracy, after every epoch will get printed to standard output. If `plot_results` is true, a plot of the learning curve will be saved in `plots/independent_sims/1250/COG28-l100-s1-aNone` after training completes. If `save_model` is true, the model will get saved in `saved_models/independent_sims/1250/COG28-l100-s1-aNone`

4. The next step is to generate embeddings of the leaf sequences from this model. The file `embeddings/config.json` specifies the dataset (MSA) and fitted model you want to generate embeddings for/from. Simply run 

```
python embeddings/gen_embeddings.py embeddings/config_gen.json
```
This will write the embeddings as a CSV file to the folder `embeddings/data/independent_sims/1250/COG28-l100-s1-aNone`

5. Now that we have embedding of the leaf sequences, we want to use a Brownian motion model to infer embeddings at the internal nodes. In other words, we want to reconstruct "ancestral embeddings". For this purpose, we use the R package **Rphylopars**, which has functionality for scalable multivariate phylogenetic comparative analysis. To obtain ancestral reconstruction of embeddings, run 

```
Rscript embeddings/embeddings_asr.R msas/independent_sims/processed/1250/COG28-l100-s1-aNone model_ld2_wd0_epoch30_2024-08-26
```

This will write a csv of ancestral embeddings into the same folder as the embeddings of the leaf sequences from step 4. 

Note that the script modifies the original tree on which the MSA evolved slightly to get rid of any tips whose sequence was an exact duplicate of another tip's sequence (such sequences had been dropped from our MSA in step 2), as well as any other external branches that are extremely short (less than 0.01). This will reduce the number of nodes in the tree, and the new size of the tree is printed to standard output. The new tree is saved to the tree directory with "_cleaned" (see step 1) replaced with "_revised" in the file name. The leaf node identitifiers (names) of this final tree are also saved to a file in the `processed` directory.

Finally, it will also produce a plot that shows the phylogenetic tree in embedding space (in the appropriate subdirectory of `plots`).

6. Last but not least, let's decode the "ancestral embeddings" back to sequences. Running 
```
python embeddings/decode_recon_embeds.py embeddings/config_decode.json 
```
will do this and evaluate the reconstructed ancestral sequences against the true ancestral sequences, calculating the identity percentage metric, and comparing it to a baseline of predicting the modal amino acid at each position for each ancestor and a baseline of IQTree using a correctly specified LG model.