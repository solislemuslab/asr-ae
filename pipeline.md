This is the entire pipeline for executing and benchmarking our method on simulated MSAs.

1. We start with the trees found at `trees/fast_trees/*/COG*.sim.trim.tree`. As discussed in `trees/fast_trees/readme.md`, these trees are taken from the FastTree study. The first step in the pipeline is to run 
```
python msas/scripts/simulate_msas.py [-l sequence length] [-s branch length scaling factor] [-a Gamma rate heterogeneity]
``` 
to simulate independent site MSAs or 
```
python msas/scripts/simulate_msas.py -type coupled [-l sequence length] [-s branch length scaling factor] [-a Gamma rate heterogeneity]
``` 
to simulate MSAs where adjacent sites are coupled.

The defaults for the three optional arguments are 100, 1, and "none", respectively. This command will generate one simulated MSA per tree in subdirectores `5k` and `1250` of `msas/independent/raw` or `msa/coupled/raw` depending on whether you simulate coupled positions or not. Note that this script will edit the trees with 5k tips before simulating the MSA to replace any negative branch lengths in the tree with branch lengths of 0.0 (the negative branch lengths arise from FastTree's algorithm). The simulated MSA files include the ancestral sequences, i.e. the sequences at the internal nodes of the tree. These internal node sequences are labelled (in what order?) with numbers that are larger than the number of tips, i.e. > 1250 if there are 1250 leaves. In contrast, the sequences at the tips have labels that begin with the letter N.

2. Next, choose an MSA to work with. Let's say I want to work with the COG28 family (with 1250 leaves) simulated with independent sites. We have to process the MSA to generate the Python objects we'll need for our Variational AutoEncoder. To do this, we run 

```
python msas/scripts/process_msa.py msas/independent/raw/1250/COG28-l100-s1.0-aNone_msa.dat 
```

This command will generate a directory in `msas/independent/processed/1250` called `COG28-l100-s1-aNone`. In this directory, you will find all the Python objects (as `.pkl` files) that represent the processed MSA for the purposes of the VAE, as well as a fasta file of the processed MSA. As opposed to the raw MSA generated in step 1, this processed MSA does not include the ancestral sequences.

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