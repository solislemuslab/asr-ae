# This code is adapted with a few changes from the PEVAE paper
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, nl, nc=21, dim_latent_vars=10, num_hidden_units=[256, 256]):
        """
        For now, we keep our model simple with both encoder and decoder having
        only fully connected layers (and the same number of them)

        Our default is that the latent embeddings are dimension 10 and there are
        256 neurons in each of the hidden layers of the encoder and decoder
        """
        super(VAE, self).__init__()

        ## num of amino acid types
        self.nc = nc

        ## length of sequences in the MSA
        self.nl = nl

        ## dimension of input
        self.dim_input = nc * nl

        ## dimension of latent space
        self.dim_latent_vars = dim_latent_vars

        ## num of hidden neurons in encoder and decoder networks
        self.num_hidden_units = num_hidden_units

        ## encoder
        self.encoder_linears = nn.ModuleList()
        self.encoder_linears.append(nn.Linear(self.dim_input, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.encoder_linears.append(nn.Linear(num_hidden_units[i-1], num_hidden_units[i]))
        self.encoder_mu = nn.Linear(num_hidden_units[-1], dim_latent_vars)
        self.encoder_logsigma = nn.Linear(num_hidden_units[-1], dim_latent_vars)

        ## decoder
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(nn.Linear(dim_latent_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_linears.append(nn.Linear(num_hidden_units[i-1], num_hidden_units[i]))
        self.decoder_linears.append(nn.Linear(num_hidden_units[-1], self.dim_input))

    def encoder(self, x):
        '''
        encoder transforms x into latent space z
        '''
        # convert from matrix to vector by concatenating rows (which are one-hot vectors)
        h = torch.flatten(x, start_dim=-2) # start_dim=-2 to maintain batch dimension
        for T in self.encoder_linears:
            h = T(h)
            h = F.relu(h)
        mu = self.encoder_mu(h)
        sigma = torch.exp(self.encoder_logsigma(h))
        return mu, sigma

    def decoder(self, z):
        '''
        decoder transforms latent space z into p, which is the probability  of x being 1.
        '''
        h = z
        for i in range(len(self.decoder_linears)-1):
            h = self.decoder_linears[i](h)
            h = F.relu(h)
        h = self.decoder_linears[-1](h) #Should now have dimension nc*nl

        fixed_shape = tuple(h.shape[0:-1])
        h = torch.unsqueeze(h, -1)
        h = torch.reshape(h, fixed_shape + (-1, self.nc))
        log_p = F.log_softmax(h, dim = -1)
        #log_p = torch.reshape(log_p, fixed_shape + (-1,))

        return log_p

    def compute_weighted_elbo(self, x, weight):
        ## sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma*eps

        ## compute log p(x|z)
        log_p = self.decoder(z)
        log_PxGz = torch.sum(x*log_p, [-1,-2]) # sum over both position and character dimension
        
        ## compute elbo
        elbo = log_PxGz - torch.sum(0.5*(sigma**2 + mu**2 - 2*torch.log(sigma) - 1), -1)
        weight = weight / torch.sum(weight)
        elbo = torch.sum(elbo*weight)

        return elbo

    def compute_elbo_with_multiple_samples(self, x, num_samples):
        '''
        Evidence lower bound is an lower bound of log P(x). Although it is a lower
        bound, we can use elbo to approximate log P(x).
        Using multiple samples to calculate the elbo makes it be a better approximation
        of log P(x).
        '''

        with torch.no_grad():
            x = x.expand(num_samples, x.shape[0], x.shape[1], x.shape[2])
            mu, sigma = self.encoder(x)
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            log_Pz = torch.sum(-0.5*z**2 - 0.5*torch.log(2*z.new_tensor(np.pi)), -1)
            log_p = self.decoder(z)
            log_PxGz = torch.sum(x*log_p, [-1,-2]) # sum over both position and character dimension
            log_Pxz = log_Pz + log_PxGz

            log_QzGx = torch.sum(-0.5*(eps)**2 -
                                 0.5*torch.log(2*z.new_tensor(np.pi))
                                 - torch.log(sigma), -1)
            log_weight = log_Pxz - log_QzGx
            log_weight = log_weight.double()
            log_weight_max = torch.max(log_weight, 0)[0]
            log_weight = log_weight - log_weight_max
            weight = torch.exp(log_weight)
            elbo = torch.log(torch.mean(weight, 0)) + log_weight_max
            return elbo
            

class LVAE(nn.Module):
    def __init__(self, nl, nc=21, dim_latent_vars=10, num_hidden_units=[256, 256]):
        """
        Our default is that the latent embeddings are dimension 10 and there are
        256 neurons in each of the hidden layers of the encoder and decoder
        """
        super(LVAE, self).__init__()

        ## num of amino acid types
        self.nc = nc

        ## length of sequences in the MSA
        self.nl = nl

        ## dimension of input
        self.dim_input = nc * nl

        ## dimension of latent space
        self.dim_latent_vars = dim_latent_vars

        ## num of hidden neurons in encoder and decoder networks
        self.num_hidden_units = num_hidden_units

        ## encoder
        self.encoder_lstm_mu = nn.LSTM(self.nc, dim_latent_vars, 1, batch_first=True)
        self.encoder_lstm_sigma = nn.LSTM(self.nc, dim_latent_vars, 1, batch_first=True)

        ## decoder
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(nn.Linear(dim_latent_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_linears.append(nn.Linear(num_hidden_units[i-1], num_hidden_units[i]))
        self.decoder_linears.append(nn.Linear(num_hidden_units[-1], self.dim_input))

    def encoder(self, x):
        '''
        encoder transforms x into latent space z
        '''
        mu, _ = self.encoder_lstm_mu(x)
        mu = mu[:, -1, :]
        sigma, _ = self.encoder_lstm_sigma(x)
        sigma = torch.exp(sigma[:, -1, :])
        return mu, sigma

    def decoder(self, z):
        '''
        decoder transforms latent space z into p, which is the probability  of x being 1.
        '''
        h = z
        for i in range(len(self.decoder_linears)-1):
            h = self.decoder_linears[i](h)
            h = F.relu(h)
        h = self.decoder_linears[-1](h) #Should now have dimension nc*nl

        fixed_shape = tuple(h.shape[0:-1])
        h = torch.unsqueeze(h, -1)
        h = torch.reshape(h, fixed_shape + (-1, self.nc))
        log_p = F.log_softmax(h, dim = -1)
        #log_p = torch.reshape(log_p, fixed_shape + (-1,))

        return log_p

    def compute_weighted_elbo(self, x, weight):
        ## sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma*eps

        ## compute log p(x|z)
        log_p = self.decoder(z)
        log_PxGz = torch.sum(x*log_p, [-1,-2]) # sum over both position and character dimension

        ## compute elbo
        elbo = log_PxGz - torch.sum(0.5*(sigma**2 + mu**2 - 2*torch.log(sigma) - 1), -1)
        weight = weight / torch.sum(weight)
        elbo = torch.sum(elbo*weight)

        return elbo

    def compute_elbo_with_multiple_samples(self, x, num_samples):
        '''
        Evidence lower bound is an lower bound of log P(x). Although it is a lower
        bound, we can use elbo to approximate log P(x).
        Using multiple samples to calculate the elbo makes it be a better approximation
        of log P(x).
        '''

        with torch.no_grad():
            x = x.expand(num_samples, x.shape[0], x.shape[1], x.shape[2])
            # LSTM doesn't take 4D input
            x = x.reshape(-1, x.shape[2], x.shape[3])
            mu, sigma = self.encoder(x)
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            log_Pz = torch.sum(-0.5*z**2 - 0.5*torch.log(2*z.new_tensor(np.pi)), -1)
            log_p = self.decoder(z)
            # log_p = log_p.reshape(x.shape)
            # sum over both position and character dimension
            log_PxGz = torch.sum(x*log_p, [-1,-2])
            # log_Pz = log_Pz.reshape(log_PxGz.shape)
            log_Pxz = log_Pz + log_PxGz

            log_QzGx = torch.sum(-0.5*(eps)**2 -
                                 0.5*torch.log(2*z.new_tensor(np.pi))
                                 - torch.log(sigma), -1)
            # log_QzGx = log_QzGx.reshape(log_Pxz.shape)
            log_weight = (log_Pxz - log_QzGx).detach().data
            log_weight = log_weight.double()
            log_weight_max = torch.max(log_weight, 0)[0]
            log_weight = log_weight - log_weight_max
            weight = torch.exp(log_weight)
            elbo = torch.log(torch.mean(weight, 0)) + log_weight_max
            return elbo


class TVAE(nn.Module):
    def __init__(
            self,nl,nc=21,
            embed_dim=8,
            num_layers=3,
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
            self.encoder_linears.append(nn.Linear(num_hidden_units[i-1], num_hidden_units[i]))
        self.encoder_mu = nn.Linear(num_hidden_units[-1], dim_latent_vars)
        self.encoder_logsigma = nn.Linear(num_hidden_units[-1], dim_latent_vars)

        # decoder
        # original layers
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(nn.Linear(dim_latent_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_linears.append(nn.Linear(num_hidden_units[i-1], num_hidden_units[i]))
        self.decoder_linears.append(nn.Linear(num_hidden_units[-1], self.dim_input))
        # transformer layers
        self.transformer_decoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.linear2 = nn.Linear(nl * embed_dim, nl * nc)


    def encoder(self, h):
        '''
        Encoder transforms x into latent space z.
        '''
        # convert from matrix to vector by concatenating rows (which are one-hot vectors)
        h = torch.flatten(h, start_dim=-2) # start_dim=-2 to maintain batch dimension
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
        '''
        Decoder transforms latent space z into p,
        which is the probability  of x being 1.
        '''
        h = z
        for i in range(len(self.decoder_linears)-1):
            h = self.decoder_linears[i](h)
            h = F.relu(h)
        h = self.decoder_linears[-1](h)  # should now have dimension embed_dim*nl

        h = h.reshape([-1, self.nl, self.embed_dim])

        h = self.transformer_decoder(h)
        h = torch.flatten(h, start_dim=-2)
        h = self.linear2(h)

        fixed_shape = tuple(h.shape[0:-1])
        h = torch.unsqueeze(h, -1)
        h = torch.reshape(h, fixed_shape + (-1, self.nc))
        log_p = F.log_softmax(h, dim = -1)
        # log_p = torch.reshape(log_p, fixed_shape + (-1,))

        return log_p

    def compute_weighted_elbo(self, x, weight):
        # sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma*eps

        # compute log p(x|z)
        log_p = self.decoder(z)
        # sum over both position and character dimension
        log_PxGz = torch.sum(x*log_p, [-1,-2])

        # compute elbo
        elbo = log_PxGz - torch.sum(0.5*(sigma**2 + mu**2 - 2*torch.log(sigma) - 1), -1)
        weight = weight / torch.sum(weight)
        elbo = torch.sum(elbo*weight)

        return elbo

    def compute_elbo_with_multiple_samples(self, x, num_samples):
        '''
        Evidence lower bound is an lower bound of log P(x). Although it is a
        lower bound, we can use elbo to approximate log P(x).
        Using multiple samples to calculate the elbo makes it be a better
        approximation of log P(x).
        '''
        with torch.no_grad():
            x = x.expand(num_samples, x.shape[0], x.shape[1], x.shape[2])
            mu, sigma = self.encoder(x)
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            log_Pz = torch.sum(-0.5*z**2 - 0.5*torch.log(2*z.new_tensor(np.pi)), -1)
            log_p = self.decoder(z)
            log_p = log_p.reshape(x.shape)
            # sum over both position and character dimension
            log_PxGz = torch.sum(x*log_p, [-1,-2])
            log_Pz = log_Pz.reshape(log_PxGz.shape)
            log_Pxz = log_Pz + log_PxGz

            log_QzGx = torch.sum(-0.5*(eps)**2 -
                                 0.5*torch.log(2*z.new_tensor(np.pi))
                                 - torch.log(sigma), -1)
            log_QzGx = log_QzGx.reshape(log_Pxz.shape)
            log_weight = log_Pxz - log_QzGx
            log_weight = log_weight.double()
            log_weight_max = torch.max(log_weight, 0)[0]
            log_weight = log_weight - log_weight_max
            weight = torch.exp(log_weight)
            elbo = torch.log(torch.mean(weight, 0)) + log_weight_max
            return elbo
