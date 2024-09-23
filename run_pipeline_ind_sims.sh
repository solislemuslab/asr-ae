#!/bin/bash

# Function to validate boolean input (only accept true or false)
validate_boolean() {
    if [[ "$1" != "true" && "$1" != "false" ]]; then
        echo "Invalid boolean value: $1. Please enter true or false."
        exit 1
    fi
}

# Function to validate numeric input
validate_number() {
    if ! [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Invalid number: $1. Please enter a valid numeric value."
        exit 1
    fi
}

# Prompt the user for input
echo "How do you want to generate the MSAs?"

echo "Enter sequence length (default 100):"
read sequence_length
sequence_length=${sequence_length:-100}  # Default to 100 if no input is provided
validate_number "$sequence_length"

echo "Enter branch length scaling factor (default 1):"
read branch_length
branch_length=${branch_length:-1}  # Default to 1 if no input is provided
validate_number "$branch_length"

echo "Enter Gamma rate heterogeneity (default 'None'):"
read gamma_rate
gamma_rate=${gamma_rate:-"None"}  # Default to 'None' if no input is provided

echo "Enter the MSA family (e.g., COG28):"
read msa_family

echo "Enter the number of sequences (e.g., 1250):"
read num_sequences

echo "Enter the query sequence (e.g., N1):"
read query_sequence
query_sequence=${query_sequence:-"N1"}  # Default to N1 if no input is provided

cd msas  # Step 1 and 2 are run from the msas directory
# Step 1: Generate MSAs
./scripts/gen_all_independent_msas.sh -l $sequence_length -s $branch_length -a $gamma_rate

# Step 2: Process MSA for the Variational AutoEncoder
python scripts/process_msa.py independent_sims/raw/$num_sequences/${msa_family}-l${sequence_length}-s${branch_length}-a${gamma_rate}_msa.dat $query_sequence --simul
echo "MSAs generated and processed successfully!"

cd ..  # Move back to the root directory
echo "How do you want to train the VAE?"

# Prompt user for the new values to update the JSON config
echo "Do you want to use transformer VAE? (true/false, default false):"
read transformer_vae
transformer_vae=${transformer_vae:-false}
validate_boolean "$transformer_vae"

echo "Enter the number of epochs (default 30):"
read epochs
epochs=${epochs:-30}
validate_number "$epochs"

echo "Enter batch size (default 32):"
read batch_size
batch_size=${batch_size:-32}
validate_number "$batch_size"

echo "Enter learning rate (default 0.001):"
read learning_rate
learning_rate=${learning_rate:-0.001}
validate_number "$learning_rate"

echo "Enter weight decay (default 0):"
read weight_decay
weight_decay=${weight_decay:-0}
validate_number "$weight_decay"

echo "Enter latent dimension (default 2):"
read latent_dim
latent_dim=${latent_dim:-2}
validate_number "$latent_dim"

echo "Verbose mode? (true/false, default true):"
read verbose
verbose=${verbose:-true}
validate_boolean "$verbose"

echo "Would you like to save the model? (true/false, default true):"
read save_model
save_model=${save_model:-true}
validate_boolean "$save_model"

echo "Would you like to plot the results? (true/false, default true):"
read plot_results
plot_results=${plot_results:-true}
validate_boolean "$plot_results"

# Path to the JSON config file
config_file="autoencoder/config.json"
tmp_file="tmp_config.json"

msa_id="${msa_family}-l${sequence_length}-s${branch_length}-a${gamma_rate}"
data_path="msas/independent_sims/processed/$num_sequences/${msa_id}"

# Use jq to update the JSON file with the user-provided values
jq  --arg MSA_id "$msa_id" \
    --arg data_path "$data_path" \
    --argjson use_transformer $transformer_vae \
    --argjson num_epochs $epochs \
    --argjson batch_size $batch_size \
    --argjson learning_rate $learning_rate \
    --argjson weight_decay $weight_decay \
    --argjson latent_dim $latent_dim \
    --argjson verbose $verbose \
    --argjson save_model $save_model \
    --argjson plot_results $plot_results \
    '.MSA_id = $MSA_id | .data_path = $data_path | .use_transformer = $use_transformer | .num_epochs = $num_epochs |
     .batch_size = $batch_size | .learning_rate = $learning_rate | .weight_decay = $weight_decay | .latent_dim = $latent_dim | 
     .verbose = $verbose | .save_model = $save_model | .plot_results = $plot_results' \
    "$config_file" | jq --indent 4 > "$tmp_file" 2> /dev/null

# Check if jq succeeded
if [ $? -ne 0 ]; then
    echo "Error: Invalid input. autoencoder/config.json not updated."
    rm -f "$tmp_file"  # Remove the temporary file if there was an error
    exit 1
fi

mv "$tmp_file" "$config_file"
echo "autoencoder/config.json updated successfully!"
echo "Training the VAE..."

# Step 3: Train the VAE
python autoencoder/train.py autoencoder/config.json

model_id="model_ld${latent_dim}_wd${weight_decay}_epoch${epochs}_$(date +%F)"
model_name="${model_id}.pt"
echo "Model Name: $model_id"
echo "Model saved at: saved_models/independent_sims/$num_sequences/${msa_id}/${model_name}"

config_file="embeddings/config_gen.json"
tmp_file="tmp_config.json"

jq  --arg MSA_id "$msa_id" \
    --arg data_path "$data_path" \
    --arg model_name "$model_name" \
    '.MSA_id = $MSA_id | .data_path = $data_path | .model_name = $model_name' \
    "$config_file" | jq --indent 4 > "$tmp_file" 2> /dev/null

# Check if jq succeeded
if [ $? -ne 0 ]; then
    echo "Error: Invalid input. embeddings/config_gen.json not updated."
    rm -f "$tmp_file"  # Remove the temporary file if there was an error
    exit 1
fi

mv "$tmp_file" "$config_file"
echo "embeddings/config_gen.json updated successfully!"
echo "Generating embeddings..."

# Step 4: Generate embeddings
python embeddings/gen_embeddings.py embeddings/config_gen.json

echo "Embeddings generated successfully!"
echo "Embeddings saved at: plots/independent_sims/$num_sequences/${msa_id}/${model_id}_embeddings.png"

echo "Inferring ancestral embeddings..."
# Step 5: Infer ancestral embeddings
Rscript embeddings/embeddings_asr.R msas/independent_sims/processed/$num_sequences/$msa_id $model_id

msa_path="msas/independent_sims/raw/$num_sequences/$msa_id"
echo "MSA path: $msa_path"
config_file="embeddings/config_decode.json"
tmp_file="tmp_config.json"

jq  --arg MSA_id "$msa_id" \
    --arg msa_path "$msa_path" \
    --arg data_path "$data_path" \
    --arg model_name "$model_name" \
    '.MSA_id = $MSA_id | .msa_path = $msa_path | .data_path = $data_path | .model_name = $model_name' \
    "$config_file" | jq --indent 4 > "$tmp_file" 2> /dev/null

# Check if jq succeeded
if [ $? -ne 0 ]; then
    echo "Error: Invalid input. embeddings/config_decode.json not updated."
    rm -f "$tmp_file"  # Remove the temporary file if there was an error
    exit 1
fi

mv "$tmp_file" "$config_file"
echo "embeddings/config_decode.json updated successfully!"

# echo "Decoding reconstructed embeddings..."
# # Step 6: Decode reconstructed embeddings
# python embeddings/decode_recon_embeds.py embeddings/config_decode.json
