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
msa_path="${data_path/processed/raw}_msa.phy"
tree_path=$(jq -r '.tree_path' $config)

########################################################
#################### Train model #######################
########################################################
# Train
echo "Training autoencoder for MSA $msa_id (data located at $data_path)"
model_name=$(python scripts/train.py config.json | tee /dev/tty | awk 'END{print}')

#######################################################
#########  Generate embeddings ########################
#######################################################
echo "Using trained VAE to generate embeddings for sequences at the tips of the tree"
plot_embeddings=$(jq '.embeddings.plot' $config)
python scripts/gen_embeddings.py $data_path $model_name --plot $plot_embeddings

#########################################################
# Reconstruct ancestral embeddings with brownian motion #
#########################################################
#TODO: change so that Rscript accepts model_name (with extension) instead of model_id (without extension)
echo "Reconstructing ancestral embeddings with brownian motion"
model_id=$(basename "$model_name" .pt)
Rscript scripts/embeddings_asr.R $data_path $model_id 

#######################################################
#########  Run ArDCA  #################################
#######################################################
echo "Running ancestral sequence reconstruction with ArDCA"
tree_path="${tree_path/trim/fully_trim}"
julia --project=. scripts/ar_reconstruct.jl $data_path $tree_path

#########################################################
## Decode to ancestral sequences and evaluate accuracy ##
#########################################################
# Update embeddings/config_decode.json
config_decode_file="embeddings/config_decode.json"
jq  --arg MSA_id "$msa_id" \
    --arg msa_path "$msa_path" \
    --arg data_path "$data_path" \
    --arg model_name "$model_name" \
    '.MSA_id = $MSA_id | .msa_path = $msa_path | .data_path = $data_path | .model_name = $model_name' \
    "$config_decode_file" | jq --indent 4 > "$TMP_FILE" 2> /dev/null
if [ $? -ne 0 ]; then
    echo "Error: Invalid input. embeddings/config_decode.json not updated."
    rm -f "$TMP_FILE"  # Remove the temporary file if there was an error
    exit 1
fi
mv $TMP_FILE $config_decode_file
echo "Decoding ancestral embeddings to sequences and evaluating all methods"
python scripts/decode_recon_embeds.py $config_decode_file



