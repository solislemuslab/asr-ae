# This code is adapted with a few changes from the PEVAE paper
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Currently VAE is a model that takes as input one hot encoded sequences, i.e. tensors of shape batch x nl x nc
# where nl is the length of the sequence and nc is the number of amino acids (21)
# On the other hand, EmbedVAE takes as input sequences of integers, i.e. tensors of shape batch x nl
# TODO: Combine the two models into one, so that the user can choose between one-hot encoding and integer encoding
# TODO: Refactor so that every specific VAE (e.g. TVAE, LVAE, etc.) is a subclass of a general VAE class


class VAE(nn.Module):
    def __init__(self, nl, nc=21,
                 dim_latent_vars=10,
                 num_hidden_units=[100], ding=False):
        """
        - nl: length of sequences in the MSA
        - nc: number of amino acid types (default is 21, which includes the gap character)
        - dim_latent_vars: dimension of latent space (default is 10)
        - num_hidden_units: list of integers representing the number of neurons in each hidden layer of the encoder and decoder networks
        - ding: If true, uses Ding's activation function (tanh) instead of ReLU

        This model accepts (batches of) sequences represented as matrices whose ith row is the one-hot encoding of the amino acid at the ith position in the sequence.
        The first thing the encoder does is concatenate the one-hot vectors for each position in the sequence into a single vector of length nl*nc.
        The rows of the weight matrix for the first encoder layer are therefore vectors representing each of the nl*nc aa-position combinations
        and our representation of each sequence (in the first hidden layer) is (the component-wise RELU of) 
        the sum of those nl rows in the weight matrix that represent aa-position combinations found in the sequence

        Similarly, the columns of the last weight matrix in the decoder are vectors representing each of the nl*nc aa-position combinations.

        Both the encoder and decoder have the same number of hidden layers and the same number of neurons in each layer.
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

        # activation function
        self.activation = F.tanh if ding else F.relu

        # encoder
        self.encoder_linears = nn.ModuleList()
        self.encoder_linears.append(
            nn.Linear(self.dim_input, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.encoder_linears.append(
                nn.Linear(num_hidden_units[i - 1], num_hidden_units[i]))
        self.encoder_mu = nn.Linear(num_hidden_units[-1], dim_latent_vars)
        self.encoder_logsigma = nn.Linear(
            num_hidden_units[-1], dim_latent_vars)

        # decoder
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(
            nn.Linear(dim_latent_vars, num_hidden_units[-1]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_linears.append(
                nn.Linear(num_hidden_units[-i], num_hidden_units[-(i+1)]))
        self.decoder_linears.append(
            nn.Linear(num_hidden_units[0], self.dim_input))

    def encoder(self, x):
        """
        encoder transforms x into latent space z
        """
        # x is batch x nl x nc
        # concatenates the nl one-hot vectors that represent each position in the sequence
        h = torch.flatten(x, start_dim=-2)
        for T in self.encoder_linears:
            h = T(h)
            h = self.activation(h)
        mu = self.encoder_mu(h)
        sigma = torch.exp(self.encoder_logsigma(h))
        return mu, sigma

    def decoder(self, z):
        """
        decoder transforms latent space z into probability distributions over amino-acids for every position in the original sequence
        """
        h = z
        for T in self.decoder_linears[:-1]:
            h = T(h)
            h = self.activation(h)
        h = self.decoder_linears[-1](h)  # batch_shape x (nl*nc)
        batch_shape = tuple(h.shape[0:-1])
        h = h.view(batch_shape + (self.nl, self.nc))  # batch_shape x nl x nc
        log_p = F.log_softmax(h, dim=-1)  # batch shape x nl x nc
        return log_p

    def forward(self, x):
        # Assumes x is batch x nl x nc
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        log_p = self.decoder(z)
        return mu, sigma, z, log_p

    def compute_weighted_elbo(self, x, weight):
        """
        Evidence lower bound is an lower bound of log P(x). Although it is a lower
        bound, we can use elbo to approximate log P(x).
        """
        # Assumes x is batch x nl x nc
        weight = weight / torch.sum(weight)

        # sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma*eps

        # compute log p(x|z)
        log_p = self.decoder(z)  # batch shape x nl x nc
        # sum over both site and character dims
        log_PxGz = torch.sum(x*log_p, [-1, -2])
        weighted_log_PxGz = log_PxGz * weight

        # compute kl
        kl = torch.sum(0.5*(sigma**2 + mu**2 - 2*torch.log(sigma) - 1), -1)
        weighted_kl = kl*weight

        # compute elbo
        weighted_elbo = weighted_log_PxGz - weighted_kl

        # return averages
        weighted_ave_elbo = torch.sum(weighted_elbo)
        weighted_ave_log_PxGz = torch.sum(weighted_log_PxGz)
        return weighted_ave_elbo, weighted_ave_log_PxGz

    @torch.no_grad()
    def compute_iwae_elbo(self, x, num_samples=100):
        """
        This is the "importance weighted" elbo, which is a different, tighter lower bound of log P(x). 
        In fact, as num_samples -> infinity, this equals log P(x).

        Note that this function does not incorporate the sequence weights. Therefore, 
        there might obtain unexpected results in comparing IWAE Elbo with regular 
        ELBO computed with non-uniform weights.
        """
        # Assumes x is batch x nl x nc
        x = x.expand(num_samples, x.shape[0], x.shape[1], x.shape[2])
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        log_Pz = torch.sum(-0.5 * z ** 2 - 0.5 *
                           torch.log(2 * z.new_tensor(np.pi)), -1)
        log_p = self.decoder(z)
        # sum over both position and character dimension
        log_PxGz = torch.sum(x * log_p, [-1, -2])
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


class EmbedVAE(nn.Module):
    def __init__(self, nl, nc=21, dim_aa_embed=5, dim_latent_vars=10, num_hidden_units=[100]):
        """
        This model accepts batches of sequences encoded as vectors of integer indices.
        The first thing the encoder does is embed each amino acid in the sequence and then concatenate the embeddings to represent the sequence

        Both the encoder and decoder have the same number of hidden layers and the same number of neurons in each layer.
        """
        super(EmbedVAE, self).__init__()

        # num of amino acid types
        self.nc = nc

        # length of sequences in the MSA
        self.nl = nl

        # dimension of latent space
        self.dim_latent_vars = dim_latent_vars

        # dimension of embedding of amino acids
        self.dim_aa_embed = dim_aa_embed

        # num of hidden neurons in encoder and decoder networks
        self.num_hidden_units = num_hidden_units

        # encoder
        self.aa_embed = nn.Embedding(nc, dim_aa_embed)
        self.encoder_linears = nn.ModuleList()
        self.encoder_linears.append(
            nn.Linear(dim_aa_embed*nl, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.encoder_linears.append(
                nn.Linear(num_hidden_units[i - 1], num_hidden_units[i]))
        self.encoder_mu = nn.Linear(num_hidden_units[-1], dim_latent_vars)
        self.encoder_logsigma = nn.Linear(
            num_hidden_units[-1], dim_latent_vars)

        # decoder
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(
            nn.Linear(dim_latent_vars, num_hidden_units[-1]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_linears.append(
                nn.Linear(num_hidden_units[-i], num_hidden_units[-(i+1)]))
        self.decoder_linears.append(nn.Linear(num_hidden_units[0], nc*nl))

    def encoder(self, x):
        """
        encoder transforms x into latent space z
        """
        # Assumes x is batch x nl
        h = self.aa_embed(x)  # batch x nl x dim_aa_embed
        # concatenates the nl embeddings of amino acids
        h = torch.flatten(h, start_dim=-2)
        for T in self.encoder_linears:
            h = T(h)
            h = F.relu(h)
        mu = self.encoder_mu(h)
        sigma = torch.exp(self.encoder_logsigma(h))
        return mu, sigma

    def decoder(self, z):
        """
        decoder transforms latent space z into probability distributions over amino-acids for every position in the original sequence
        """
        h = z
        for T in self.decoder_linears[:-1]:
            h = T(h)
            h = F.relu(h)
        h = self.decoder_linears[-1](h)  # batch_shape x (nl*nc)
        batch_shape = tuple(h.shape[0:-1])
        h = h.view(batch_shape + (self.nl, self.nc))  # batch_shape x nl x nc
        log_p = F.log_softmax(h, dim=-1)  # batch shape x nl x nc
        return log_p

    def forward(self, x):
        # Assumes x is batch x nl
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        log_p = self.decoder(z)
        return mu, sigma, z, log_p

    def compute_weighted_elbo(self, x, weight):
        """
        Evidence lower bound is an lower bound of log P(x). Although it is a lower
        bound, we can use elbo to approximate log P(x).
        """
        # Assumes x is batch x nl
        weight = weight / torch.sum(weight)

        # sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma*eps

        # compute log p(x|z)
        log_p = self.decoder(z)  # (batch shape x nl x nc)
        # we want to index log_p (batch shape x nl x nc) with x (batch_shape x nl)
        log_PxGz = torch.gather(
            log_p, -1, x.unsqueeze(-1)).squeeze()  # batch_shape x nl
        log_PxGz = torch.sum(log_PxGz, -1)  # sum over sites
        weighted_log_PxGz = log_PxGz * weight

        # compute kl
        kl = torch.sum(0.5*(sigma**2 + mu**2 - 2*torch.log(sigma) - 1), -1)
        weighted_kl = kl*weight

        # compute elbo
        weighted_elbo = weighted_log_PxGz - weighted_kl

        # return averages
        weighted_ave_elbo = torch.sum(weighted_elbo)
        weighted_ave_log_PxGz = torch.sum(weighted_log_PxGz)
        return weighted_ave_elbo, weighted_ave_log_PxGz

    @torch.no_grad()
    def compute_iwae_elbo(self, x, num_samples=100):
        """
        This is the "importance weighted" elbo, which is a different, tighter lower bound of log P(x). 
        In fact, as num_samples -> infinity, this equals log P(x).

        Note that this function does not incorporate the sequence weights. Therefore, 
        there might obtain unexpected results in comparing IWAE Elbo with regular 
        ELBO computed with non-uniform weights.
        """
        x = x.expand(num_samples, x.shape[0], x.shape[1])
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        log_Pz = torch.sum(-0.5 * z ** 2 - 0.5 *
                           torch.log(2 * z.new_tensor(np.pi)), -1)
        log_p = self.decoder(z)
        log_PxGz = torch.gather(
            log_p, -1, x.unsqueeze(-1)).squeeze()  # batch_shape x nl
        log_PxGz = torch.sum(log_PxGz, -1)  # sum over sites
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
        mu, _ = self.encoder(x)
        log_p = self.decoder(mu)
        pred_aa_idxs = torch.argmax(log_p, -1)
        recon_acc = torch.mean((x == pred_aa_idxs).float(), -1)
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
        self.encoder_lstm_mu = nn.LSTM(
            self.nc, dim_latent_vars, 1, batch_first=True)
        self.encoder_lstm_sigma = nn.LSTM(
            self.nc, dim_latent_vars, 1, batch_first=True)

        # decoder
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(
            nn.Linear(dim_latent_vars, num_hidden_units[-1]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_linears.append(
                nn.Linear(num_hidden_units[-i], num_hidden_units[-(i+1)]))
        self.decoder_linears.append(
            nn.Linear(num_hidden_units[0], self.dim_input))

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
        h = self.decoder_linears[-1](h)  # batch_shape x (nl*nc)
        batch_shape = tuple(h.shape[0:-1])
        h = h.view(batch_shape + (self.nl, self.nc))  # batch_shape x nl x nc
        log_p = F.log_softmax(h, dim=-1)  # batch shape x nl x nc
        return log_p

    def compute_weighted_elbo(self, x, weight):
        """
        Evidence lower bound is an lower bound of log P(x). Although it is a lower
        bound, we can use elbo to approximate log P(x).
        """
        weight = weight / torch.sum(weight)

        # sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma*eps

        # compute log p(x|z)
        log_p = self.decoder(z)
        # sum over both site and character dims
        log_PxGz = torch.sum(x*log_p, [-1, -2])
        weighted_log_PxGz = log_PxGz * weight

        # compute kl
        kl = torch.sum(0.5*(sigma**2 + mu**2 - 2*torch.log(sigma) - 1), -1)
        weighted_kl = kl*weight

        # compute elbo
        weighted_elbo = weighted_log_PxGz - weighted_kl

        # return averages
        weighted_ave_elbo = torch.sum(weighted_elbo)
        weighted_ave_log_PxGz = torch.sum(weighted_log_PxGz)
        return weighted_ave_elbo, weighted_ave_log_PxGz

    @torch.no_grad()
    def compute_iwae_elbo(self, x, num_samples):
        """
        This is the "importance weighted" elbo, which is a different, tighter lower bound of log P(x). 
        In fact, as num_samples -> infinity, this equals log P(x).

        Note that this function does not incorporate the sequence weights. Therefore, 
        there might obtain unexpected results in comparing IWAE Elbo with regular 
        ELBO computed with non-uniform weights.
        """

        x = x.expand(num_samples, x.shape[0], x.shape[1], x.shape[2])
        # LSTM doesn't take 4D input
        x = x.reshape(-1, x.shape[2], x.shape[3])
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        log_Pz = torch.sum(-0.5 * z ** 2 - 0.5 *
                           torch.log(2 * z.new_tensor(np.pi)), -1)
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
        self.encoder_linears.append(
            nn.Linear(self.dim_input, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.encoder_linears.append(
                nn.Linear(num_hidden_units[i - 1], num_hidden_units[i]))
        self.encoder_mu = nn.Linear(num_hidden_units[-1], dim_latent_vars)
        self.encoder_logsigma = nn.Linear(
            num_hidden_units[-1], dim_latent_vars)

        # decoder
        # original layers
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(
            nn.Linear(dim_latent_vars, num_hidden_units[-1]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_linears.append(
                nn.Linear(num_hidden_units[-i], num_hidden_units[-(i+1)]))
        self.decoder_linears.append(
            nn.Linear(num_hidden_units[0], self.dim_input))
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
        # start_dim=-2 to maintain batch dimension
        h = torch.flatten(h, start_dim=-2)
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
        # should now have dimension embed_dim*nl
        h = self.decoder_linears[-1](h)

        h = h.reshape([-1, self.nl, self.embed_dim])

        h = self.transformer_decoder(h)
        h = torch.flatten(h, start_dim=-2)
        h = self.linear2(h)

        batch_shape = tuple(h.shape[0:-1])
        h = h.view(batch_shape + (self.nl, self.nc))  # batch_shape x nl x nc
        log_p = F.log_softmax(h, dim=-1)  # batch shape x nl x nc
        return log_p

    def compute_weighted_elbo(self, x, weight):
        """
        Evidence lower bound is an lower bound of log P(x). Although it is a lower
        bound, we can use elbo to approximate log P(x).
        """
        weight = weight / torch.sum(weight)

        # sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma*eps

        # compute log p(x|z)
        log_p = self.decoder(z)
        # sum over both site and character dims
        log_PxGz = torch.sum(x*log_p, [-1, -2])
        weighted_log_PxGz = log_PxGz * weight

        # compute kl
        kl = torch.sum(0.5*(sigma**2 + mu**2 - 2*torch.log(sigma) - 1), -1)
        weighted_kl = kl*weight

        # compute elbo
        weighted_elbo = weighted_log_PxGz - weighted_kl

        # return averages
        weighted_ave_elbo = torch.sum(weighted_elbo)
        weighted_ave_log_PxGz = torch.sum(weighted_log_PxGz)
        return weighted_ave_elbo, weighted_ave_log_PxGz

    @torch.no_grad()
    def compute_iwae_elbo(self, x, num_samples):
        """
        This is the "importance weighted" elbo, which is a different, tighter lower bound of log P(x). 
        In fact, as num_samples -> infinity, this equals log P(x).

        Note that this function does not incorporate the sequence weights. Therefore, 
        there might obtain unexpected results in comparing IWAE Elbo with regular 
        ELBO computed with non-uniform weights.
        """
        x = x.expand(num_samples, x.shape[0], x.shape[1], x.shape[2])
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        log_Pz = torch.sum(-0.5 * z ** 2 - 0.5 *
                           torch.log(2 * z.new_tensor(np.pi)), -1)
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
