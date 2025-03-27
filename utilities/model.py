import torch
from autoencoder.model import VAE

def load_model(model_path, nl, nc=21, num_hidden_units=[256, 256], nlatent=2):
    """
    Load the model from the model path.
    TODO: Allow loading transformer or LSTM models
    """
    model = VAE(nl = nl, nc = nc, num_hidden_units=num_hidden_units, dim_latent_vars=nlatent) 
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)
    return model
