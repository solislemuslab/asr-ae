# This code is adapted with a few changes from the PEVAE paper
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


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
            

class LSTM_VAE(nn.Module):
    def __init__(self, nl, nc=21, dim_latent_vars=10, num_hidden_units=[256, 256]):
        """
        Our default is that the latent embeddings are dimension 10 and there are
        256 neurons in each of the hidden layers of the encoder and decoder
        """
        super(LSTM_VAE, self).__init__()

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
            # x = x.expand(num_samples, x.shape[0], x.shape[1], x.shape[2])
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
            log_weight = (log_Pxz - log_QzGx).detach().data
            log_weight = log_weight.double()
            log_weight_max = torch.max(log_weight, 0)[0]
            log_weight = log_weight - log_weight_max
            weight = torch.exp(log_weight)
            elbo = torch.log(torch.mean(weight, 0)) + log_weight_max
            return elbo


class Tokenizer:
    def __init__(self):
        # special tokens
        vocab = ["<cls>", "<pad>", "<eos>", "<unk>"]
        # 20 anonical amino acids
        vocab += list("ACDEFGHIKLMNPQRSTVWY")
        # mapping
        self.token_to_index = {tok: i for i, tok in enumerate(vocab)}
        self.index_to_token = {i: tok for i, tok in enumerate(vocab)}

    @property
    def vocab_size(self):
        return len(self.token_to_index)

    @property
    def pad_token_id(self):
        return self.token_to_index["<pad>"]

    def __call__(
            self, seqs: List[str], padding: bool = True
        ) -> Dict[str, List[List[int]]]:
        """
        Tokenizes a list of protein sequences and
        creates input representations with attention masks.
        """

        input_ids = []
        attention_mask = []

        if padding:
            max_len = max(len(seq) for seq in seqs)

        for seq in seqs:
            # Preprocessing: strip whitespace, convert to uppercase
            seq = seq.strip().upper()

            # Add special tokens
            toks = ["<cls>"] + list(seq) + ["<eos>"]

            if padding:
                # Pad with '<pad>' tokens to reach max_len
                toks += ["<pad>"] * (max_len - len(seq))

            # Convert tokens to IDs (handling unknown amino acids)
            unk_id = self.token_to_index["<unk>"]
            input_ids.append(
                [self.token_to_index.get(tok, unk_id) for tok in toks]
            )

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask.append([1 if tok != "<pad>" else 0 for tok in toks])

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class PositionalEncoding(nn.Module):
    """
    Injects positional information into token embeddings using sine and cosine.

    The positional encoding is added to the token embeddings to provide the
    model with information about the position of each token in a sequence.

    Args:
        embedding_dim (int): The dimension of the token embeddings.
        max_length (int, optional): The maximum sequence length for which positional
            encodings will be pre-computed. Defaults to 5000.

    Attributes:
        pe (torch.Tensor): A pre-computed positional encoding tensor of shape
            (1, max_length, embedding_dim).
    """

    def __init__(self, embedding_dim: int, max_length: int = 5000) -> None:
        super().__init__()

        pe = torch.zeros(max_length, embedding_dim)

        # add an addtitional dimention for broadcasting
        position = torch.arange(max_length).float().unsqueeze(1)

        # div_term is of length embedding_dim//2:
        div_term = torch.exp(
            - torch.arange(0, embedding_dim, 2) / embedding_dim * np.log(1e4)
        )

        # populate even and odd indices
        # position*div_term: (max_length, embedding_dim//2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # reshape for broadcasting:
        # (max_length, embedding_dim) => (1, max_length, embedding_dim)
        pe = pe.unsqueeze(0)

        # pe is not a parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        Adds positional encodings to input embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length,
            embedding_dim).

        Returns:
            torch.Tensor: Input tensor with positional encodings added,
            of the same shape.
        """

        return x + self.pe[:, : x.shape[1], :]  # (batch, seq_len, embedding_dim)


class TransformerVAE(nn.Module):
    def __init__(
            self,
            vocab_size,
            seq_length,
            embedding_dim,
            num_layers,
            dim_latent_vars=10,
            num_hidden_units=[256, 256]
        ):
        """
        Variational autoencoder with transformer/attention layers.
        """
        super(TransformerVAE, self).__init__()
        self.embedding_dim = embedding_dim
        self.dim_input = embedding_dim * seq_length

        # embedding layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.positional_encoding = PositionalEncoding(embedding_dim, seq_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            batch_first=True,
            activation="gelu"
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=8,
            batch_first=True,
            activation="gelu"
        )

        # encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.encoder_linears = nn.ModuleList()
        self.encoder_linears.append(nn.Linear(self.dim_input, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.encoder_linears.append(nn.Linear(num_hidden_units[i-1], num_hidden_units[i]))
        self.encoder_mu = nn.Linear(num_hidden_units[-1], dim_latent_vars)
        self.encoder_logsigma = nn.Linear(num_hidden_units[-1], dim_latent_vars)

        self.memory = None

        # decoder layers
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(nn.Linear(dim_latent_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_linears.append(nn.Linear(num_hidden_units[i-1], num_hidden_units[i]))
        self.decoder_linears.append(nn.Linear(num_hidden_units[-1], self.dim_input))
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

    def embed(self, x):
        """
        Embed sequences using token embeddings and positional encodings.
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)
        return x

    def encoder(self, x):
        """
        Encode embeddings into latent space.
        """
        x = self.transformer_encoder(x)
        self.memory = x
        h = torch.flatten(x, start_dim=-2)  # maintain batch dimension
        for T in self.encoder_linears:
            h = T(h)
            h = F.relu(h)
        mu = self.encoder_mu(h)
        sigma = torch.exp(self.encoder_logsigma(h))
        return mu, sigma
        
    def decoder(self, z):
        """
        Decode latent space into embeddings.
        """
        h = z
        for i in range(len(self.decoder_linears) - 1):
            h = self.decoder_linears[i](h)
            h = F.relu(h)
        h = self.decoder_linears[-1](h)  # should now have dimension dim_input

        fixed_shape = tuple(h.shape[0:-1])
        h = torch.unsqueeze(h, -1)
        h = torch.reshape(h, fixed_shape + (-1, self.embedding_dim))
        h = self.transformer_decoder(h, self.memory)

        return h

    def loss(self, x):
        """
        Compute the loss function.
        """
        x = self.embed(x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        x_hat = self.decoder(z)

        # reconstruction loss
        loss_recon = F.mse_loss(x_hat, x)

        # KL divergence
        kl_div = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

        return loss_recon, kl_div
