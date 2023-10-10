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

def train(
    model: nn.Module,
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.StepLR] = None,
    vae_type: int = 1,
    print_iter: int = 10,
    ckpt_path: str = None,
) -> Tuple[List[nn.Module], List[int]]:
    """Training loop.
    
    Parameters
    ----------
    model : nn.Module
        The model to train.
    epochs : int
        The number of training epochs.
    train_loader : torch.utils.data.DataLoader
        The data loader with training data.
    optimizer : torch.optim.Optimizer
        The optimizer.
    scheduler : Optional[torch.optim.lr_scheduler.StepLR]
        The learning rate scheduler. I suggest a step scheduler, but you can use any.
    vae_type : int, default=1
        Type of vae. Choices: {1, 2}
    print_iter : int, default=10
        Iterations between training printouts.
    ckpt_path : str, default=None
        Path to a checkpoint to load to continue training.
        
    Returns
    -------
    model_list : List[nn.Module]
        A list of snapshots of the model at each epoch of training.
    loss_list : List[int]
        A list of losses of the model at each epoch of training.
    """
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
        print('Loaded Checkpoint!')
    model.train()  # training mode
    
    iteration = 1
    model_list = []
    loss_list = []
    
    for epoch in range(epochs):
        for batch_idx, minibatch in enumerate(train_loader):
            optimizer.zero_grad()  # reset gradients

            # check model type
            if vae_type == 1: 
                true_d = minibatch[:,:model._obs_dim]
                conditioning_vars = minibatch[:,model._obs_dim:]
                mu_e, var_e, mu_p, var_p, mu_d, var_d = model(true_d, conditioning_vars)
                loss = model.loss( true_d, mu_d, var_d, mu_e, var_e, mu_p, var_p)

                # if torch.isnan(loss): 
                #     breakpoint()
                
            else:
                raise NotImplementedError

            # Backpropagation
            loss.backward()


            # Perform gradient clipping with a max_norm of 1.0
            max_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


            # Update the model parameters
            optimizer.step()    
                        
            # printing training information every 100 iterations
            if iteration % print_iter == 0:
                print(f"Epoch: {epoch + 1}\tIteration: {iteration}\tLoss: {loss.item():.4f}")
            
            iteration = iteration + 1
            
        # advance the lr scheduler
        if scheduler is not None:
            scheduler.step()
        
        # update cached performance of the model
        model_list.append(copy.deepcopy(model))
        loss_list.append(loss.item())
    
    return model_list, loss_list