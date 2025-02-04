# This code is adapted with a few changes from the PEVAE paper
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Refactor so that every specific VAE is a subclass of a general VAE class


def load_model(model_path, nl, nc, num_hidden_units=[256, 256], nlatent=2):
    """
    Load the model from the model path.
    TODO: Allow loading transformer or LSTM models
    """
    model = VAE(nl = nl, nc = nc, num_hidden_units=num_hidden_units, dim_latent_vars=nlatent) 
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)
    return model


class VAE(nn.Module):
    def __init__(self, nl, nc=21, dim_latent_vars=10, num_hidden_units=[256, 256]):
        """
        For now, we keep our model simple with both encoder and decoder having
        only fully connected layers (and the same number of them)

        Our default is that the latent embeddings are dimension 10 and there are
        256 neurons in each of the hidden layers of the encoder and decoder
        """
        super(VAE, self).__init__()

        # num of amino acid types
        self.nc = nc

        # length of sequences in the MSA
        self.nl = nl

        # dimension of input
        self.dim_input = nc * nl

        # dimension of latent space
        self.dim_latent_vars = dim_latent_vars

        # num of hidden neurons in encoder and decoder networks
        self.num_hidden_units = num_hidden_units

        # encoder
        self.encoder_linears = nn.ModuleList()
        self.encoder_linears.append(nn.Linear(self.dim_input, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.encoder_linears.append(nn.Linear(num_hidden_units[i - 1], num_hidden_units[i]))
        self.encoder_mu = nn.Linear(num_hidden_units[-1], dim_latent_vars)
        self.encoder_logsigma = nn.Linear(num_hidden_units[-1], dim_latent_vars)

        # decoder
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(nn.Linear(dim_latent_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_linears.append(nn.Linear(num_hidden_units[i - 1], num_hidden_units[i]))
        self.decoder_linears.append(nn.Linear(num_hidden_units[-1], self.dim_input))

    def encoder(self, x):
        """
        encoder transforms x into latent space z
        """
        h = torch.flatten(x, start_dim=-2)  # concatenates the one-hot vectors
        for T in self.encoder_linears:
            h = T(h)
            h = F.relu(h)
        mu = self.encoder_mu(h)
        sigma = torch.exp(self.encoder_logsigma(h))
        return mu, sigma

    def decoder(self, z):
        """
        decoder transforms latent space z into probability distributions over amino-acids for every position in seq
        """
        h = z
        for i in range(len(self.decoder_linears) - 1):
            h = self.decoder_linears[i](h)
            h = F.relu(h)
        h = self.decoder_linears[-1](h) # batch_shape x (nl*nc) 
        batch_shape = tuple(h.shape[0:-1]) 
        h = h.view(batch_shape + (self.nl, self.nc)) # batch_shape x nl x nc
        log_p = F.log_softmax(h, dim=-1) # batch shape x nl x nc 
        return log_p

    def compute_weighted_elbo(self, x, weight):
        weight = weight / torch.sum(weight)
        
        ## sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma*eps

        ## compute log p(x|z)
        log_p = self.decoder(z) # batch shape x nl x nc
        log_PxGz = torch.sum(x*log_p, [-1, -2]) # sum over both site and character dims
        weighted_log_PxGz = log_PxGz * weight

        ## compute kl
        kl =  torch.sum(0.5*(sigma**2 + mu**2 - 2*torch.log(sigma) - 1), -1)
        weighted_kl = kl*weight

        ## compute elbo
        weighted_elbo = weighted_log_PxGz - weighted_kl
        
        ## return averages
        weighted_ave_elbo = torch.sum(weighted_elbo)
        weighted_ave_log_PxGz = torch.sum(weighted_log_PxGz)
        return weighted_ave_elbo, weighted_ave_log_PxGz

    @torch.no_grad()
    def compute_iwae_elbo(self, x, num_samples=100):
        """
        Evidence lower bound is an lower bound of log P(x). Although it is a lower
        bound, we can use elbo to approximate log P(x).
        Using multiple samples to calculate the elbo makes it be a better approximation
        of log P(x).

        Note that this function does not incorporate the sequence weights. Therefore, 
        there might obtain unexpected results in comparing IWAE Elbo with regular 
        ELBO computed with non-uniform weights.
        """
        x = x.expand(num_samples, x.shape[0], x.shape[1], x.shape[2])
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        log_Pz = torch.sum(-0.5 * z ** 2 - 0.5 * torch.log(2 * z.new_tensor(np.pi)), -1)
        log_p = self.decoder(z)
        log_PxGz = torch.sum(x * log_p, [-1, -2])  # sum over both position and character dimension
        log_Pxz = log_Pz + log_PxGz
        log_QzGx = torch.sum(-0.5 * (eps) ** 2 -
                             0.5 * torch.log(2 * z.new_tensor(np.pi))
                             - torch.log(sigma), -1)
        log_weight = log_Pxz - log_QzGx
        log_weight = log_weight.double()
        log_weight_max = torch.max(log_weight, 0)[0]
        log_weight = log_weight - log_weight_max
        weight = torch.exp(log_weight)
        elbo = torch.log(torch.mean(weight, 0)) + log_weight_max
        return elbo
    
    @torch.no_grad()
    def compute_acc(self, x):
        '''
        Calculates the Hamming accuracy (i.e. percent residue identity)
        '''
        real_aa_idxs = torch.argmax(x, -1)
        mu, _ = self.encoder(x)
        log_p = self.decoder(mu)
        pred_aa_idxs = torch.argmax(log_p, -1)
        recon_acc = torch.mean((real_aa_idxs == pred_aa_idxs).float(), -1)
        return recon_acc

class LVAE(nn.Module):
    def __init__(self, nl, nc=21, dim_latent_vars=10, num_hidden_units=[256, 256]):
        """
        Our default is that the latent embeddings are dimension 10 and there are
        256 neurons in each of the hidden layers of the encoder and decoder
        """
        super(LVAE, self).__init__()

        # num of amino acid types
        self.nc = nc

        # length of sequences in the MSA
        self.nl = nl

        # dimension of input
        self.dim_input = nc * nl

        # dimension of latent space
        self.dim_latent_vars = dim_latent_vars

        # num of hidden neurons in encoder and decoder networks
        self.num_hidden_units = num_hidden_units

        # encoder
        self.encoder_lstm_mu = nn.LSTM(self.nc, dim_latent_vars, 1, batch_first=True)
        self.encoder_lstm_sigma = nn.LSTM(self.nc, dim_latent_vars, 1, batch_first=True)

        # decoder
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(nn.Linear(dim_latent_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_linears.append(nn.Linear(num_hidden_units[i - 1], num_hidden_units[i]))
        self.decoder_linears.append(nn.Linear(num_hidden_units[-1], self.dim_input))

    def encoder(self, x):
        """
        encoder transforms x into latent space z
        """
        mu, _ = self.encoder_lstm_mu(x)
        mu = mu[:, -1, :]
        sigma, _ = self.encoder_lstm_sigma(x)
        sigma = torch.exp(sigma[:, -1, :])
        return mu, sigma

    def decoder(self, z):
        """
        decoder transforms latent space z into probability distributions over amino-acids for every position in seq
        """
        h = z
        for i in range(len(self.decoder_linears) - 1):
            h = self.decoder_linears[i](h)
            h = F.relu(h)
        h = self.decoder_linears[-1](h) # batch_shape x (nl*nc) 
        batch_shape = tuple(h.shape[0:-1]) 
        h = h.view(batch_shape + (self.nl, self.nc)) # batch_shape x nl x nc
        log_p = F.log_softmax(h, dim=-1) # batch shape x nl x nc 
        return log_p

    def compute_weighted_elbo(self, x, weight):
        weight = weight / torch.sum(weight)
        
        ## sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma*eps

        ## compute log p(x|z)
        log_p = self.decoder(z)
        log_PxGz = torch.sum(x*log_p, [-1, -2]) # sum over both site and character dims
        weighted_log_PxGz = log_PxGz * weight

        ## compute kl
        kl =  torch.sum(0.5*(sigma**2 + mu**2 - 2*torch.log(sigma) - 1), -1)
        weighted_kl = kl*weight

        ## compute elbo
        weighted_elbo = weighted_log_PxGz - weighted_kl
        
        ## return averages
        weighted_ave_elbo = torch.sum(weighted_elbo)
        weighted_ave_log_PxGz = torch.sum(weighted_log_PxGz)
        return weighted_ave_elbo, weighted_ave_log_PxGz

    @torch.no_grad()
    def compute_iwae_elbo(self, x, num_samples):
        """
        Evidence lower bound is an lower bound of log P(x). Although it is a lower
        bound, we can use elbo to approximate log P(x).
        Using multiple samples to calculate the elbo makes it be a better approximation
        of log P(x).
        """

        x = x.expand(num_samples, x.shape[0], x.shape[1], x.shape[2])
        # LSTM doesn't take 4D input
        x = x.reshape(-1, x.shape[2], x.shape[3])
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        log_Pz = torch.sum(-0.5 * z ** 2 - 0.5 * torch.log(2 * z.new_tensor(np.pi)), -1)
        log_p = self.decoder(z)
        # log_p = log_p.reshape(x.shape)
        # sum over both position and character dimension
        log_PxGz = torch.sum(x * log_p, [-1, -2])
        # log_Pz = log_Pz.reshape(log_PxGz.shape)
        log_Pxz = log_Pz + log_PxGz
        log_QzGx = torch.sum(-0.5 * (eps) ** 2 -
                             0.5 * torch.log(2 * z.new_tensor(np.pi))
                             - torch.log(sigma), -1)
        # log_QzGx = log_QzGx.reshape(log_Pxz.shape)
        log_weight = (log_Pxz - log_QzGx).detach().data
        log_weight = log_weight.double()
        log_weight_max = torch.max(log_weight, 0)[0]
        log_weight = log_weight - log_weight_max
        weight = torch.exp(log_weight)
        elbo = torch.log(torch.mean(weight, 0)) + log_weight_max
        return elbo
    
    @torch.no_grad()
    def compute_acc(self, x):
        '''
        Calculates the Hamming accuracy (i.e. percent residue identity)
        '''
        real_aa_idxs = torch.argmax(x, -1)
        mu, _ = self.encoder(x)
        log_p = self.decoder(mu)
        pred_aa_idxs = torch.argmax(log_p, -1)
        recon_acc = torch.mean((real_aa_idxs == pred_aa_idxs).float(), -1)
        return recon_acc

class TVAE(nn.Module):
    def __init__(
            self, nl, nc=21,
            embed_dim=8,
            num_layers=1,
            dim_latent_vars=10,
            num_hidden_units=[256, 256]
    ):
        super(TVAE, self).__init__()

        # num of amino acid types
        self.nc = nc
        # length of sequences in the MSA
        self.nl = nl
        # embedding dimension
        self.embed_dim = embed_dim
        # dimension of input
        self.dim_input = embed_dim * nl
        # dimension of latent space
        self.dim_latent_vars = dim_latent_vars
        # num of hidden neurons in encoder and decoder networks
        self.num_hidden_units = num_hidden_units

        # encoder
        # transformer layers
        self.linear1 = nn.Linear(nl * nc, nl * embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # original layers
        self.encoder_linears = nn.ModuleList()
        self.encoder_linears.append(nn.Linear(self.dim_input, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.encoder_linears.append(nn.Linear(num_hidden_units[i - 1], num_hidden_units[i]))
        self.encoder_mu = nn.Linear(num_hidden_units[-1], dim_latent_vars)
        self.encoder_logsigma = nn.Linear(num_hidden_units[-1], dim_latent_vars)

        # decoder
        # original layers
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(nn.Linear(dim_latent_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_linears.append(nn.Linear(num_hidden_units[i - 1], num_hidden_units[i]))
        self.decoder_linears.append(nn.Linear(num_hidden_units[-1], self.dim_input))
        # transformer layers
        self.transformer_decoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.linear2 = nn.Linear(nl * embed_dim, nl * nc)

    def encoder(self, h):
        """
        Encoder transforms x into latent space z.
        """
        # convert from matrix to vector by concatenating rows (which are one-hot vectors)
        h = torch.flatten(h, start_dim=-2)  # start_dim=-2 to maintain batch dimension
        h = self.linear1(h)
        h = h.reshape([-1, self.nl, self.embed_dim])
        h = self.transformer_encoder(h)

        h = torch.flatten(h, start_dim=-2)
        for T in self.encoder_linears:
            h = T(h)
            h = F.relu(h)
        mu = self.encoder_mu(h)
        sigma = torch.exp(self.encoder_logsigma(h))
        return mu, sigma

    def decoder(self, z):
        """
        Decoder transforms latent space z into p,
        which is the probability  of x being 1.
        """
        h = z
        for i in range(len(self.decoder_linears) - 1):
            h = self.decoder_linears[i](h)
            h = F.relu(h)
        h = self.decoder_linears[-1](h)  # should now have dimension embed_dim*nl

        h = h.reshape([-1, self.nl, self.embed_dim])

        h = self.transformer_decoder(h)
        h = torch.flatten(h, start_dim=-2)
        h = self.linear2(h)

        batch_shape = tuple(h.shape[0:-1]) 
        h = h.view(batch_shape + (self.nl, self.nc)) # batch_shape x nl x nc
        log_p = F.log_softmax(h, dim=-1) # batch shape x nl x nc 
        return log_p

    def compute_weighted_elbo(self, x, weight):
        weight = weight / torch.sum(weight)
        
        ## sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma*eps

        ## compute log p(x|z)
        log_p = self.decoder(z)
        log_PxGz = torch.sum(x*log_p, [-1, -2]) # sum over both site and character dims
        weighted_log_PxGz = log_PxGz * weight

        ## compute kl
        kl =  torch.sum(0.5*(sigma**2 + mu**2 - 2*torch.log(sigma) - 1), -1)
        weighted_kl = kl*weight

        ## compute elbo
        weighted_elbo = weighted_log_PxGz - weighted_kl
        
        ## return averages
        weighted_ave_elbo = torch.sum(weighted_elbo)
        weighted_ave_log_PxGz = torch.sum(weighted_log_PxGz)
        return weighted_ave_elbo, weighted_ave_log_PxGz
    
    @torch.no_grad()
    def compute_iwae_elbo(self, x, num_samples):
        """
        Evidence lower bound is an lower bound of log P(x). Although it is a
        lower bound, we can use elbo to approximate log P(x).
        Using multiple samples to calculate the elbo makes it be a better
        approximation of log P(x).
        """
        x = x.expand(num_samples, x.shape[0], x.shape[1], x.shape[2])
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        log_Pz = torch.sum(-0.5 * z ** 2 - 0.5 * torch.log(2 * z.new_tensor(np.pi)), -1)
        log_p = self.decoder(z)
        log_p = log_p.reshape(x.shape)
        # sum over both position and character dimension
        log_PxGz = torch.sum(x * log_p, [-1, -2])
        log_Pz = log_Pz.reshape(log_PxGz.shape)
        log_Pxz = log_Pz + log_PxGz
        log_QzGx = torch.sum(-0.5 * (eps) ** 2 -
                             0.5 * torch.log(2 * z.new_tensor(np.pi))
                             - torch.log(sigma), -1)
        log_QzGx = log_QzGx.reshape(log_Pxz.shape)
        log_weight = log_Pxz - log_QzGx
        log_weight = log_weight.double()
        log_weight_max = torch.max(log_weight, 0)[0]
        log_weight = log_weight - log_weight_max
        weight = torch.exp(log_weight)
        elbo = torch.log(torch.mean(weight, 0)) + log_weight_max
        return elbo
    
    @torch.no_grad()
    def compute_acc(self, x):
        '''
        Calculates the Hamming accuracy (i.e. percent residue identity)
        '''  
        real_aa_idxs = torch.argmax(x, -1)
        mu, _ = self.encoder(x)
        log_p = self.decoder(mu)
        pred_aa_idxs = torch.argmax(log_p, -1)
        recon_acc = torch.mean((real_aa_idxs == pred_aa_idxs).float(), -1)
        return recon_acc
