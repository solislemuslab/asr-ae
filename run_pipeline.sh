#!/bin/bash
TMP_FILE="tmp_config.json"

# main config file
if [ -z "$1" ]; then
  config="config.json"
else
  config=$1
fi
# family information
data_path=$(jq -r '.data_path' $config)
msa_id=$(basename "$data_path")
msa_path=$(jq -r '.msa_path' $config)
tree_path=$(jq -r '.tree_path' $config)

########################################################
#################### Train model #######################
########################################################
# Train
echo "Training autoencoder for MSA $msa_id (data located at $data_path)"
model_name=$(python scripts/train.py $config | tee /dev/tty | awk 'END{print}')

#######################################################
#########  Generate embeddings ########################
#######################################################
# Update config.json for decoding
jq --arg model_name "$model_name" \
    '.generate.model_name = $model_name | .generate.model_gapped_data_not = true' \
    $config | jq --indent 4 > $TMP_FILE 2> /dev/null
if [ $? -ne 0 ]; then
    echo "Error: Invalid input. config.json generate parameters not updated."
    rm -f "$TMP_FILE"  # Remove the temporary file if there was an error
    exit 1
fi
mv $TMP_FILE $config
echo "Using trained VAE to generate embeddings for sequences at the tips of the tree"
python scripts/gen_embeddings.py $config

#########################################################
# Reconstruct ancestral embeddings with brownian motion #
#########################################################
Rscript scripts/embeddings_asr.R $data_path $model_name 

#######################################################
#########  Run ArDCA  #################################
#######################################################
echo "Running ancestral sequence reconstruction with ArDCA"
tree_path="${tree_path/trim/fully_trim}"
julia --project=. scripts/ar_reconstruct.jl $data_path $tree_path

#########################################################
## Decode to ancestral sequences and evaluate accuracy ##
#########################################################
# Update config.json for decoding
jq  --arg model_names "$model_name" \
    --arg plot_name "${model_name/.pt/_eval.png}" \
    '.decode.model_names = [$model_names] | .decode.plot_name = $plot_name' \
    $config | jq --indent 4 > "$TMP_FILE" 2> /dev/null
if [ $? -ne 0 ]; then
    echo "Error: Invalid input. config.json decoding parameters not updated."
    rm -f "$TMP_FILE"  # Remove the temporary file if there was an error
    exit 1
fi
mv $TMP_FILE $config
echo "Decoding ancestral embeddings to sequences and evaluating all methods"
python scripts/decode_recon_embeds.py $config



