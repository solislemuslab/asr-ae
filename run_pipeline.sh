#!/bin/bash
TMP_FILE="tmp_config.json"

#######################################################
###### Retrieve values from main config file ##########
#######################################################
if [ -z "$1" ]; then
  config="config.json"
else
  config=$1
fi
# family information
msa_id=$(jq -r '.MSA_id' $config)
data_path=$(jq -r '.data_path' $config)
msa_path="${data_path/processed/raw}_msa.dat"
# model information
latent_dim=$(jq -r '.latent_dim' $config)
layers=$(jq -r '.num_hidden_units | join("-")' $config)
weight_decay=$(jq -r '.weight_decay' $config)
epochs=$(jq -r '.num_epochs' $config)
model_id="model_layers${layers}_ld${latent_dim}_wd${weight_decay}_epoch${epochs}_$(date +%F)"
model_name="${model_id}.pt"

########################################################
#################### Train model #######################
########################################################
echo "Training autoencoder for MSA $msa_id (data located at $data_path)"
python train.py config.json

#######################################################
#########  Generate embeddings ########################
#######################################################
# Update embeddings/config_gen.json
config_gen_file="embeddings/config_gen.json"
jq  --arg MSA_id "$msa_id" \
    --arg data_path "$data_path" \
    --arg model_name "$model_name" \
    '.MSA_id = $MSA_id | .data_path = $data_path | .model_name = $model_name' \
    "$config_gen_file" | jq --indent 4 > "$TMP_FILE" 2> /dev/null
if [ $? -ne 0 ]; then
    echo "Error: Invalid input. embeddings/config_gen.json not updated."
    rm -f "$TMP_FILE"  # Remove the temporary file if there was an error
    exit 1
fi
mv $TMP_FILE $config_gen_file
python embeddings/gen_embeddings.py $config_gen_file

#########################################################
# Reconstruct ancestral embeddings with brownian motion #
#########################################################
Rscript embeddings/embeddings_asr.R $data_path $model_id

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
python embeddings/decode_recon_embeds.py $config_decode_file



