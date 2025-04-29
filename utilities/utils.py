import os
import re
def get_directory(data_path, folder, data_subfolder = False):
    """
    Take data path and MSA id and return the directory where the data is stored.
    """
    MSA_id = os.path.basename(data_path)
    if MSA_id[0:3] == "COG": # this is a simulated dataset
        sim_type = os.path.dirname(data_path).split("/")[1] 
        num_seqs = os.path.dirname(data_path).split("/")[-1] 
        dir =  f"{folder}/{sim_type}/{num_seqs}/{MSA_id}"
    else:
        dir = f"{folder}/real/{MSA_id}"
    if data_subfolder:
        dir_list = dir.split("/")
        dir_list.insert(1, "data")
        dir = ("/").join(dir_list)
    return dir

def parse_model_name(model_name):

    # Latent dimension of VAE
    # use re to check if model name starts with "trans"
    if re.match(r'^trans', model_name):
        is_transformer = True
    else:
        is_transformer = False
    ld = int(re.search(r'ld(\d+)', model_name).group(1))
    # Number of hidden units in VAE
    layers_match = re.search(r'layers(\d+(\-\d+)*)', model_name)
    num_hidden_units = [int(size) for size in layers_match.group(1).split('-')]
    # aa embedding dimension will be present in the model name if model is an EmbedVAE
    aa_embed_match = re.search(r'aaembed(\d+)', model_name)
    dim_aa_embed = int(aa_embed_match.group(1)) if aa_embed_match else None
    one_hot = not aa_embed_match

    return is_transformer, ld, num_hidden_units, dim_aa_embed, one_hot