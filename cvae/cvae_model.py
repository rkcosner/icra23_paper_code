import copy
from math import pi as pi
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision.datasets import MNIST as MNIST
from typing import List, Optional, Tuple
import time
import pdb


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.set_default_device(device)

class MLP(nn.Module):
    """Helper class for initializing multilayer perceptrons."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_sizes: List[int],
        nonlinearity: nn.Module,
        dropout: Optional[float] = None,
        batchnorm: Optional[bool] = False,
    ) -> None:
        """Initialize the MLP. Activations are softpluses.

        Parameters
        ----------
        input_dim : int
            Dimension of the input.
        output_dim : int
            Dimension of the output variable.
        hidden_layer_sizes : List[int]
            List of sizes of all hidden layers.
        nonlinearity : torch.nn.Module
            A the nonlinearity to use (must be a torch module).
        dropout : float, default=None
            Dropout probability if applied.
        batchnorm : bool, default=False
            Flag for applying batchnorm. NOTE: there are a plethora of ways
            to apply batchnorm layers. I chose post-activations.
        """
        super(MLP, self).__init__()

        assert type(input_dim) == int
        assert type(output_dim) == int
        assert type(hidden_layer_sizes) == list
        assert all(type(n) is int for n in hidden_layer_sizes)

        # building MLP
        self._mlp = nn.Sequential()
        self._mlp.add_module("fc0", nn.Linear(input_dim, hidden_layer_sizes[0]))
        self._mlp.add_module("act0", nonlinearity)
        if dropout is not None and 0.0 <= dropout and dropout <= 1.0:
            self._mlp.add_module("do0", nn.Dropout(p=dropout))
        if batchnorm:
            self._mlp.add_module("bn0", nn.BatchNorm1d(hidden_layer_sizes[0]))
        for i, (in_size, out_size) in enumerate(
            zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:]), 1
        ):
            self._mlp.add_module(f"fc{i}", nn.Linear(in_size, out_size))
            self._mlp.add_module(f"act{i}", nonlinearity)
            if dropout is not None and 0.0 <= dropout and dropout <= 1.0:
                self._mlp.add_module("do{i}", nn.Dropout(p=dropout))
            if batchnorm:
                self._mlp.add_module("bn{i}", nn.BatchNorm1d(out_size))
        self._mlp.add_module("fcout", nn.Linear(hidden_layer_sizes[-1], output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape=(..., in_dim)
            Dimension of the input.

        Returns
        -------
        mlp_out : torch.Tensor, shape=(..., out_dim)
            Output tensor.
        """
        return self._mlp(x)



###########################3
#############################3

def gaussian_log_likelihood(mu: torch.Tensor, var: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluates the log likelihood of diagonal-covariance Gaussian parameters given data.
    
    Parameters
    ----------
    mu : torch.Tensor, shape=(B, X)
        Batch of means of Gaussians.
    var : torch.Tensor, shape=(B, X)
        Batch of (diagonal) covariances of Gaussians.
    x : torch.Tensor, shape=(B, X)
        Batch of data.
        
    Returns
    -------
    ll : torch.Tensor, shape=(1)
        Batch-averaged log likelihood of the parameters.
    """
    B, X = x.shape
    ll = -torch.sum(0.5 * (X * torch.log(2 * torch.Tensor([pi])).to(device) + torch.log(var) + (x - mu) ** 2 / var)) / B

    return ll

def gaussian_dkl(
    mu_1: torch.Tensor,
    var_1: torch.Tensor,
    mu_2: torch.Tensor = None,
    var_2: torch.Tensor = None
) -> torch.Tensor:
    """Computes the analytical KL divergence between two diagonal-covariance Gaussians.
    
    Consider two Gaussian distributions D_1 = N(mu_1, var_1) and D_2 = N(mu_2, var_2).
    This function will compute D_KL(D_1 || D_2). If the parameters of D_2 are none,
    then D_2 is assumed to be the standard normal distribution.
    
    Parameters
    ----------
    mu_1 : torch.Tensor, shape=(B, X)
        Mean of D_1.
    var_1 : torch.Tensor, shape=(B, X)
        Diagonal entries of covariance of D_1.
    mu_2 : torch.Tensor, shape=(B, X), default=None
        Mean of D_2. Optional.
    var_2 : torch.Tensor, shape=(B, X), default=None
        Diagonal entries of covariance of D_2. Optional.
        
    Returns
    -------
    dkl : torch.Tensor, shape=(1)
        The batch-averged KL divergence.
    """
    B, X = mu_1.shape
    if mu_2 is None or var_2 is None:
        dkl = 0.5 * torch.sum(torch.sum(-torch.log(var_1) + var_1 + mu_1 ** 2, dim=-1) - X) / B
    else:
        dkl = 0.5 * torch.sum(
            torch.sum(torch.log(var_2) - torch.log(var_1) + (var_1 + (mu_2 - mu_1) ** 2) / var_2, dim=-1) - X
        ) / B
    return dkl

#####################
#####################

class CVAE(nn.Module):
    """Conditional Variational Autoencoder (CVAE)."""
    
    def __init__(self, latent_dim: int, obs_dim: int, input_dim: int, hidden_layer_sizes: int= 16) -> None:
        """Initialize a CVAE for modeling LxL single channel images.
        
        Parameters
        ----------
        latent_dim : int
            Dimension of the latent space. Denoted n in the code.
        obs_dim : int
            Dimension of the observation space. Denoted p in the code.
            NOTE: we assume the data is flattened before encoding and
            decoding will return flattened data as well.
        input_dim : int
            Dimension of the conditioning input. Denoted m in the code.
        """
        super(CVAE, self).__init__()
        
        # assigning dimensions
        self._latent_dim = latent_dim
        self._obs_dim = obs_dim
        self._input_dim = input_dim

        # initializing networks
        self._encoder = MLP(
            obs_dim + input_dim,
            2 * latent_dim,
            hidden_layer_sizes,
            nn.Softplus(beta=2),
        )
        self._decoder = MLP(
                latent_dim + input_dim,
                2 * obs_dim,
                hidden_layer_sizes,
                nn.Softplus(beta=2),
        )
        
        self._prior = MLP(
            input_dim,
            2 * latent_dim,
            hidden_layer_sizes,
            nn.Softplus(beta=2),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the VAE.
        
        Parameters
        ----------
        x : torch.Tensor, shape=(B, C, sqrt(p), sqrt(p))
            Batch of input data (channel images).
        c : torch.Tensor, shape=(B, m)
            Batch of conditioning variables (integers in [0, 9]).
            
        Returns
        -------
        mu_e : torch.Tensor, shape=(B, n)
            Batch of means of encodings of x.
        var_e : torch.Tensor, shape=(B, n)
            Batch of (diagonal) covariances of encodings of x.
        mu_p : torch.Tensor, shape=(B, n)
            Batch of means of conditional latent prior.
        var_p : torch.Tensor, shape=(B, n)
            Batch of (diagonal) covariances of conditional latent prior.
        x_recon_flat : torch.Tensor, shape=(B, p)
            Batch of flattened reconstructions.
        """
        # shapes
        n = self._latent_dim
        p = self._obs_dim
        m = self._input_dim
        
        # prior
        out_p = self._prior(c)
        mu_p = out_p[..., :n]
        var_p = torch.exp(out_p[..., n:]) + 1e-6  # regularization
        
        # encoder
        out_e = self._encoder(torch.cat([x.reshape(-1, p), c], dim=-1))
        mu_e = out_e[..., :n]
        var_e = torch.exp(out_e[..., n:]) + 1e-6  # regularization
        
        # reparameterization
        z = self.reparameterize(mu_e, var_e)
        
        # decoder
        out_d = self._decoder(torch.cat([z, c], dim=-1))
        mu_d = out_d[..., :p]
        var_d = torch.exp(out_d[..., p:]) + 1e-6

        return mu_e, var_e, mu_p, var_p, mu_d, var_d 
    
    def loss(
        self,
        true_d: torch.Tensor, 
        mu_d: torch.Tensor, 
        var_d: torch.Tensor, 
        mu_e: torch.Tensor,
        var_e: torch.Tensor,
        mu_p: torch.Tensor,
        var_p: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss.
        
        Parameters
        ----------
        See docstring for forward().
        
        Returns
        -------
        loss : torch.Tensor, shape=(1)
            The VAE loss.
        """

        loss_recon = -gaussian_log_likelihood(mu_d, var_d, true_d)
        loss_dkl = gaussian_dkl(mu_e, var_e, mu_p, var_p)
        loss = loss_recon + loss_dkl
        return loss
    
    def reparameterize(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Execute the reparameterization trick for diagonal-covariance Gaussian.
        
        Parameters
        ----------
        mu : torch.Tensor, shape=(B, X)
            Batch of input means.
        var : torch.Tensor, shape=(B, X)
            Batch of input (diagonal) covariances.
            
        Returns
        -------
        y : torch.Tensor, shape=(B, X)
            Sample from N(mu, var)
        """
        eps = torch.randn_like(var)
        y = mu + torch.sqrt(var) * eps  # element-wise because of diagonal covariance
        return y
    
    def generate(self, num_samples: int, c: torch.Tensor) -> torch.Tensor:
        """Generate data.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        c : torch.Tensor, shape=(B, m)
            Batch of conditioning variables (integers in [0, 9]).
        
        Returns
        -------
        x_gen : torch.Tensor, shape=(num_samples, sqrt(p), sqrt(p))
            Generated image data of proper shape.
        """
        # shapes
        n = self._latent_dim
        p = self._obs_dim
        L = int(np.sqrt(p))

        # drawing a sample from the latent prior
        # out_p = self._prior(c)
        # mu_p = out_p[..., :n]
        # var_p = torch.exp(out_p[..., n:]) + 1e-6  # regularization
        # z_gen = self.reparameterize(mu_p, var_p)

        cs = c.repeat(num_samples,1)
        out_p = self._prior(cs)
        mu_p = out_p[..., :n]
        var_p = torch.exp(out_p[..., n:]) + 1e-6  # regularization
        z_gen = self.reparameterize(mu_p, var_p)

        # decoding the random sample
        out_d  = self._decoder(torch.cat([z_gen, cs], dim=-1))
        # mu_d = out_d[..., :n]
        # var_d = torch.exp(out_d[..., n:]) + 1e-6
        # if num_samples == 1:
        #     x_gen = x_gen_flat.reshape(L, L)
        # else:
        #     x_gen = x_gen_flat.reshape(-1, L, L)
            
        mu_gen = out_d[..., :p]
        var_gen = torch.exp(out_d[..., p:])
        eps = torch.randn_like(var_gen)
        x_gen = mu_gen + torch.sqrt(var_gen) * eps


        return x_gen

    def get_mean_cov(self, num_samples: int, c: torch.Tensor) -> torch.Tensor:
        """Generate data.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        c : torch.Tensor, shape=(1, m)
            Single conditioning variable 

        Returns
        -------
        mean_gen : torch.Tensor, shape=(p,)
        cov_gen  : torch.Tensor, shape=(p,) 
        """

        # shapes
        n = self._latent_dim
        p = self._obs_dim
        L = int(np.sqrt(p))

        # drawing a sample from the latent prior
        cs = c.repeat(num_samples,1)
        out_p = self._prior(cs)
        mu_p = out_p[..., :n]
        var_p = torch.exp(out_p[..., n:]) + 1e-6  # regularization

        # create samples
        z_gen = self.reparameterize(mu_p, var_p)
        

        # decoding the random sample
        out_d  = self._decoder(torch.cat([z_gen, cs], dim=-1))
        
        mus = out_d[..., :p]
        mu_gen = torch.sum(out_d[..., :p], axis = 0 ) / num_samples
        mu_gen = mu_gen.reshape((p,1))

        # start = time.time()
        # cov_gen = - mu_gen @ mu_gen.T
        # for n in range(num_samples): 
        #     cov = torch.diag(torch.exp(out_d[n, p:])) 
        #     mu = mus[n,:].reshape((p,1)) 
        #     cov_gen += (cov + mu @ mu.T ) / num_samples
        # print(time.time() - start)

        # start = time.time()
        # orders of magnitude faster, use batch summing instead of for looping
        cov_gen = torch.exp(out_d[:,p:]) 
        cov_gen = torch.diag(torch.sum(cov_gen, axis=0)) / num_samples- mu_gen @ mu_gen.T
        mus = mus.unsqueeze(-1)
        cov_gen += torch.sum(torch.matmul(mus, mus.transpose(1,2)), axis=0) / num_samples
        # print(time.time() - start)
        # breakpoint()

        return mu_gen, cov_gen

    def get_mean_cov_and_rescale(self, num_samples: int, c: torch.Tensor, target_scaler) -> torch.Tensor:
        """Generate data.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        c : torch.Tensor, shape=(1, m)
            Single conditioning variable 

        Returns
        -------
        mean_gen : torch.Tensor, shape=(p,)
        cov_gen  : torch.Tensor, shape=(p,) 
        """

        # shapes
        n = self._latent_dim
        p = self._obs_dim
        L = int(np.sqrt(p))

        # drawing a sample from the latent prior
        cs = c.repeat(num_samples,1)
        out_p = self._prior(cs)
        mu_p = out_p[..., :n]
        var_p = torch.exp(out_p[..., n:]) + 1e-6  # regularization

        # create samples
        z_gen = self.reparameterize(mu_p, var_p)
        
        # decoding the random sample
        out_d  = self._decoder(torch.cat([z_gen, cs], dim=-1))

        # time_old = time.time()
        mus = out_d[..., :p] * target_scaler.scale_ + target_scaler.mean_
        # print(time.time() - time_old)
        # breakpoint()
        mu_gen = torch.sum(mus, axis = 0 ) / num_samples
        mu_gen = mu_gen.reshape((p,1))
        cov_gen = - mu_gen @ mu_gen.T

        # for n in range(num_samples): 
        #     cov = torch.diag(torch.exp(out_d[n, p:])) * torch.diag(torch.tensor(target_scaler.var_)) 
        #     mu = mus[n,:].reshape((p,1))
        #     cov_gen += (cov + mu @ mu.T ) / num_samples
        cov_gen = torch.exp(out_d[:,p:]) * target_scaler.var_
        cov_gen = torch.diag(torch.sum(cov_gen, axis=0)) / num_samples- mu_gen @ mu_gen.T
        mus = mus.unsqueeze(-1)
        cov_gen += torch.sum(torch.matmul(mus, mus.transpose(1,2)), axis=0) / num_samples

        return mu_gen, cov_gen

    def generate_and_rescale(self, num_samples: int, c: torch.Tensor, target_scaler) -> torch.Tensor:
        """Generate data.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        c : torch.Tensor, shape=(B, m)
            Batch of conditioning variables (integers in [0, 9]).
        
        Returns
        -------
        x_gen : torch.Tensor, shape=(num_samples, sqrt(p), sqrt(p))
            Generated image data of proper shape.
        """
        # shapes
        n = self._latent_dim
        p = self._obs_dim
        L = int(np.sqrt(p))

        # drawing a sample from the latent prior
        out_p = self._prior(c)
        mu_p = out_p[..., :n]
        var_p = torch.exp(out_p[..., n:]) + 1e-6  # regularization
        z_gen = self.reparameterize(mu_p, var_p)
        
        # decoding the random sample
        out_d  = self._decoder(torch.cat([z_gen, c], dim=-1))
        # mu_d = out_d[..., :n]
        # var_d = torch.exp(out_d[..., n:]) + 1e-6
        # if num_samples == 1:
        #     x_gen = x_gen_flat.reshape(L, L)
        # else:
        #     x_gen = x_gen_flat.reshape(-1, L, L)

        if len(out_d.shape) == 1: 
            out_d = out_d.reshape((1, out_d.shape[0]))
        mu_gen = out_d[..., :p] * target_scaler.scale_ + target_scaler.mean_
        var_gen = torch.exp(out_d[..., p:]) * target_scaler.var_
        eps = torch.randn_like(var_gen)
        x_gen = mu_gen + torch.sqrt(var_gen) * eps


        return x_gen[0]