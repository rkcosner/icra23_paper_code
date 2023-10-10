import torch
from torch import nn
import diffusion_models
import diffusion_utils

# Implement MLP for forward pass.
class DIDiffusionMLP(nn.Module):
    def __init__(self, time_emb_dim, state_dim, cond_emb_dim):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.state_dim = state_dim
        self.cond_emb_dim = cond_emb_dim
        # self.net = diffusion_models.MLP(
        #     input_dim=self.state_dim + time_emb_dim + cond_emb_dim,
        #     hidden_dims=[32, 32, 32, 32],
        #     output_dim=self.state_dim,
        # )
        emb_dim = time_emb_dim + cond_emb_dim
        self.layers = nn.ModuleList([
            nn.Linear(self.state_dim + time_emb_dim + cond_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256+ emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256 + emb_dim, 256),
            nn.ReLU(),
            # nn.Linear(32 + emb_dim, 32),
            # nn.ReLU(),
            nn.Linear(256 + emb_dim, self.state_dim),
        ])
        # self.net = nn.Sequential(
        #     nn.Linear(self.state_dim + time_emb_dim + cond_emb_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, self.state_dim),
        # )

    def forward(self, x, conditioning):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = torch.cat([x, conditioning], dim=-1)
            x = layer(x)
        return x
        # x = torch.cat([x, conditioning], dim=-1)
        # return self.net(x)
