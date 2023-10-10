import torch
import torch.nn as nn
from functools import partial
from typing import List, Tuple
import math

from enum import Enum, auto


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build model
        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(
            zip([input_dim] + list(hidden_dims[:-1]), hidden_dims)
        ):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.SiLU())
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        assert x.shape[-1] == self.input_dim, f"{x.shape} != {self.input_dim}"
        for layer in self.layers:
            x = layer(x)
        return x


class FiLMLayer(nn.Module):
    """
    A PyTorch implementation of a FiLM layer.
    """

    def __init__(
        self,
        num_output_dims: int,
        in_channels: int,
        conditioning_dim: int,
        hidden_dims: Tuple[int, ...] = (32,),
    ):
        super().__init__()
        self.num_output_dims = num_output_dims
        self.in_channels = in_channels
        self.conditioning_dim = conditioning_dim
        self.hidden_dims = hidden_dims

        # Build model
        self.gamma = MLP(
            conditioning_dim, hidden_dims, in_channels
        )  # Map conditioning dimension to scaling for each channel.
        self.beta = MLP(conditioning_dim, hidden_dims, in_channels)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        # Compute FiLM parameters
        assert conditioning.shape[-1] == self.conditioning_dim
        batch_dims = conditioning.shape[:-1]

        assert x.shape[: -(self.num_output_dims + 1)] == batch_dims
        assert x.shape[-(self.num_output_dims + 1)] == self.in_channels

        gamma = self.gamma(conditioning)
        beta = self.beta(conditioning)

        # Do unsqueezing to make sure dimensions match; run e.g., twice for 2D FiLM.
        for _ in range(self.num_output_dims):
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        assert (
            gamma.shape
            == beta.shape
            == batch_dims + (self.in_channels,) + (1,) * self.num_output_dims
        )

        # Apply FiLM
        return (1 + gamma) * x + beta


class CNN2DFiLM(nn.Module):
    """
    A vanilla 2D CNN with FiLM conditioning layers
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        conv_channels: List[int],
        conditioning_dim: int,
        num_in_channels: int,
        pooling=nn.MaxPool2d(kernel_size=2),
        film_hidden_layers: Tuple[int, ...] = (32,),
        dropout_every: int = 1,
        pooling_every: int = 2,
        condition_every: int = 1,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.conv_channels = conv_channels
        self.conditioning_dim = conditioning_dim  # Note conditioning can only be 1D.
        self.num_in_channels = num_in_channels
        self.pooling = pooling
        self.film_hidden_layers = film_hidden_layers
        self.dropout_every = dropout_every
        self.pooling_every = pooling_every
        self.condition_every = condition_every

        # Build model
        self.conv_layers = nn.ModuleList()
        for i, (in_channels, out_channels) in enumerate(
            zip([self.num_in_channels] + conv_channels[:-1], conv_channels)
        ):
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.conv_layers.append(nn.SiLU())
            if i % self.dropout_every == 0:
                self.conv_layers.append(nn.Dropout2d(p=0.1, inplace=False))
            self.conv_layers.append(nn.BatchNorm2d(out_channels))
            if i % self.condition_every == 0:
                self.conv_layers.append(
                    FiLMLayer(
                        2,
                        out_channels,
                        conditioning_dim,
                        hidden_dims=film_hidden_layers,
                    )
                )
            if i % self.pooling_every == 0:
                self.conv_layers.append(self.pooling)

        # Compute output shape
        with torch.no_grad():
            self.output_shape = self.get_output_shape()

    def get_output_shape(self):
        # Compute output shape
        x = torch.zeros((1, self.num_in_channels, *self.input_shape))
        x = self.forward(x, torch.zeros((1, self.conditioning_dim)))

        return x.shape[1:]

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        """
        Forward pass for FiLM 2D CNN.

        Args:
            x: input tensor of shape (batch_size, num_in_channels, *input_shape)
            conditioning: conditioning tensor of shape (batch_size, conditioning_dim)
        """
        assert x.shape[-2:] == self.input_shape
        assert x.shape[-3] == self.num_in_channels
        assert conditioning.shape[-1] == self.conditioning_dim

        # Reshape batch dims
        if len(x.shape) >= 4:
            batch_dims = x.shape[:-3]
            assert batch_dims == conditioning.shape[:-1]

            x = x.reshape(-1, *x.shape[-3:])
            conditioning = conditioning.reshape(-1, *conditioning.shape[-1:])
        else:
            batch_dims = None

        for layer in self.conv_layers:
            if isinstance(layer, FiLMLayer):
                x = layer(x, conditioning)
            else:
                x = layer(x)

        # Reshape batch dims
        if batch_dims is not None:
            x = x.reshape(*batch_dims, *x.shape[-3:])  # Batch dims, n_c, n_w, n_h

        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(in_channels, num_heads=num_heads)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size, channels, h, w = x.shape

        # Reshape x to (height*width, batch_size, channels)
        x = x.permute(2, 3, 0, 1).contiguous().view(h * w, batch_size, channels)

        # Apply self-attention
        attn_output, _ = self.multihead_attn(x, x, x)

        # Reshape back to (batch_size, channels, height, width)
        attn_output = attn_output.view(h, w, batch_size, channels).permute(2, 3, 0, 1)

        # Combine attention output with input through a convolutional layer
        x = self.conv(attn_output)
        x = self.norm(x)
        x = self.relu(x)

        return x


class ContractBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conditioning_dim, film_hidden_layers):
        super(ContractBlock, self).__init__()
        self.conditioning_dim = conditioning_dim
        self.film_hidden_layers = film_hidden_layers

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.film1 = FiLMLayer(
            2,
            out_channels,
            self.conditioning_dim,
            hidden_dims=self.film_hidden_layers,
        )
        self.group_norm1 = nn.GroupNorm(4, out_channels)
        self.relu1 = nn.SiLU(inplace=True)
        self.self_attention1 = SelfAttention(out_channels, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.group_norm2 = nn.GroupNorm(8, out_channels)
        self.film2 = FiLMLayer(
            2,
            out_channels,
            self.conditioning_dim,
            hidden_dims=self.film_hidden_layers,
        )
        self.relu2 = nn.SiLU(inplace=True)
        self.self_attention2 = SelfAttention(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_conv1 = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, conditioning):
        h = self.conv1(x)
        h = self.group_norm1(h)
        h = self.film1(h, conditioning)
        h = self.relu1(h)

        # Residual connection
        x = self.res_conv1(x) + h
        x = self.self_attention1(x)

        h = self.conv2(x)
        h = self.group_norm2(h)
        h = self.film2(h, conditioning)
        h = self.relu2(h)
        x = x + h
        x = self.self_attention2(x)
        x = self.pool(x)

        return x


class ExpandBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conditioning_dim, film_hidden_layers):
        super(ExpandBlock, self).__init__()
        self.conditioning_dim = conditioning_dim
        self.film_hidden_layers = film_hidden_layers

        self.conv1 = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.film1 = FiLMLayer(
            2,
            out_channels,
            self.conditioning_dim,
            hidden_dims=self.film_hidden_layers,
        )
        self.relu1 = nn.SiLU(inplace=True)
        self.self_attention1 = SelfAttention(
            out_channels, out_channels, num_heads=4 if out_channels > 1 else 1
        )

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.film2 = FiLMLayer(
            2,
            out_channels,
            self.conditioning_dim,
            hidden_dims=self.film_hidden_layers,
        )
        self.relu2 = nn.SiLU(inplace=True)
        self.self_attention2 = SelfAttention(
            out_channels, out_channels, num_heads=4 if out_channels > 1 else 1
        )

        self.group_norm1 = (
            nn.GroupNorm(16, out_channels) if out_channels > 1 else nn.Identity()
        )
        self.group_norm2 = (
            nn.GroupNorm(8, out_channels) if out_channels > 1 else nn.Identity()
        )

        self.res_conv1 = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, conditioning):
        h = self.conv1(x)
        h = self.group_norm1(h)
        h = self.film1(h, conditioning)
        h = self.relu1(h)

        # Residual connection
        x = self.res_conv1(x) + h
        x = self.self_attention1(x)

        h = self.conv2(x)
        h = self.group_norm2(h)
        h = self.film2(h, conditioning)
        h = self.relu2(h)
        x = x + h
        x = self.self_attention2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear")

        return x


class ConditionedUNetModel(nn.Module):
    def __init__(self, in_channels, out_channels, conditioning_dim, film_hidden_layers):
        super(ConditionedUNetModel, self).__init__()
        self.conditioning_dim = conditioning_dim
        self.film_hidden_layers = film_hidden_layers

        # Contracting path
        self.encoder1 = ContractBlock(
            in_channels, 32, conditioning_dim, film_hidden_layers
        )
        self.encoder2 = ContractBlock(32, 64, conditioning_dim, film_hidden_layers)

        # Bottleneck (no pooling or downsampling)
        self.middle = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

        # Expansive path
        self.decoder2 = ExpandBlock(128, 32, conditioning_dim, film_hidden_layers)
        self.decoder1 = ExpandBlock(
            64, out_channels, conditioning_dim, film_hidden_layers
        )

        self.output = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x, conditioning):
        # Contracting path
        x1 = self.encoder1(x, conditioning)
        x2 = self.encoder2(x1, conditioning)

        # Middle (bottleneck) path
        x3 = self.middle(x2)

        # Expansive path with residual connections
        x4 = self.decoder2(torch.cat([x2, x3], dim=1), conditioning)
        x5 = self.decoder1(torch.cat([x1, x4], dim=1), conditioning)

        x_out = self.output(x)

        return x_out


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CNN2DFiLM(nn.Module):
    """
    A vanilla 2D CNN with FiLM conditioning layers
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        conv_channels: List[int],
        conditioning_dim: int,
        num_in_channels: int,
        film_hidden_layers: Tuple[int, ...] = (32,),
        dropout_every: int = 1,
        condition_every: int = 1,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.conv_channels = conv_channels
        self.conditioning_dim = conditioning_dim  # Note conditioning can only be 1D.
        self.num_in_channels = num_in_channels
        self.film_hidden_layers = film_hidden_layers
        self.dropout_every = dropout_every
        self.condition_every = condition_every

        # Build model
        self.conv_layers = nn.ModuleList()
        for i, (in_channels, out_channels) in enumerate(
            zip([self.num_in_channels] + conv_channels[:-1], conv_channels)
        ):
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.conv_layers.append(nn.ReLU())
            if i % self.dropout_every == 0:
                self.conv_layers.append(nn.Dropout2d(p=0.1, inplace=False))
            self.conv_layers.append(nn.BatchNorm2d(out_channels))
            if i % self.condition_every == 0:
                self.conv_layers.append(
                    FiLMLayer(
                        2,
                        out_channels,
                        conditioning_dim,
                        hidden_dims=film_hidden_layers,
                    )
                )

        # Compute output shape
        with torch.no_grad():
            self.output_shape = self.get_output_shape()

    def get_output_shape(self):
        # Compute output shape
        x = torch.zeros((1, self.num_in_channels, *self.input_shape))
        x = self.forward(x, torch.zeros((1, self.conditioning_dim)))

        return x.shape[1:]

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        """
        Forward pass for FiLM 2D CNN.

        Args:
            x: input tensor of shape (batch_size, num_in_channels, *input_shape)
            conditioning: conditioning tensor of shape (batch_size, conditioning_dim)
        """
        assert x.shape[-2:] == self.input_shape
        assert x.shape[-3] == self.num_in_channels
        assert conditioning.shape[-1] == self.conditioning_dim

        # Reshape batch dims
        if len(x.shape) >= 4:
            batch_dims = x.shape[:-3]
            assert batch_dims == conditioning.shape[:-1]

            x = x.reshape(-1, *x.shape[-3:])
            conditioning = conditioning.reshape(-1, *conditioning.shape[-1:])
        else:
            batch_dims = None

        for layer in self.conv_layers:
            if isinstance(layer, FiLMLayer):
                x = layer(x, conditioning)
            else:
                x = layer(x)

        # Reshape batch dims
        if batch_dims is not None:
            x = x.reshape(*batch_dims, *x.shape[-3:])  # Batch dims, n_c, n_w, n_h

        return x


class DiffusionModel(nn.Module):
    def __init__(
        self, num_channels, num_time_embeddings, time_emb_dim, film_hidden_layers
    ):
        super(DiffusionModel, self).__init__()
        # self.model = ConditionedUNetModel(
        #     num_channels, num_channels, time_emb_dim, film_hidden_layers
        # )
        self.model = CNN2DFiLM(
            (28, 28),
            [32, 64, 128, 256],
            time_emb_dim,
            num_channels,
            film_hidden_layers=film_hidden_layers,
        )

        self.output_layer = nn.Sequential(
            SelfAttention(self.model.conv_channels[-1], self.model.conv_channels[-1]),
            nn.ReLU(),
            nn.Conv2d(self.model.conv_channels[-1], num_channels, kernel_size=1),
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(num_time_embeddings),
            nn.Linear(num_time_embeddings, time_emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, x, t):
        time_emb = self.time_mlp(t)
        x = self.model(x, time_emb)
        return self.output_layer(x)

    def set_beta_schedule(self, timesteps, beta_schedule):
        self.timesteps = timesteps
        self.register_buffer("betas", beta_schedule(timesteps))
        self.register_buffer("alpha_bars", (1 - self.betas).cumprod(0))
        self.register_buffer("alphas", 1 - self.betas)

        self.register_buffer(
            "alpha_bars_prev",
            torch.nn.functional.pad(self.alpha_bars[:-1], (1, 0), value=1.0),
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            "posterior_vars",
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars),
        )

    def get_beta(self, step):
        return self.betas[step]

    def forward_sample(self, x0, t, eps=None):
        """Sample from a single diffusion path starting at x0 at time t."""
        assert x0.shape[0] == t.shape[0]

        assert len(x0.shape) == 4

        # Sample noise
        if eps is None:
            eps = torch.randn_like(x0)

        # Sample from model
        alpha_bar = (
            self.alpha_bars[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )  # Unsqueeze so image shaped.

        x = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * eps

        return x

    def get_loss(self, x0):
        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)

        # Denoise images
        if len(x0.shape) == 3:
            x0 = x0.unsqueeze(1)

        # Sample random noise
        eps = torch.randn_like(x0)

        # Sample noisy images
        x = self.forward_sample(x0, t, eps=eps)

        noise_pred = self.forward(x, t)

        # Compute loss
        loss = nn.functional.smooth_l1_loss(noise_pred, eps)
        # loss = torch.mean(noise_pred**2)

        return loss


class ConditionalDiffusionCNN(nn.Module):
    def __init__(
        self,
        num_channels,
        conditioning_dim,
        film_hidden_layers,
    ):
        super(ConditionalDiffusionCNN, self).__init__()
        # self.model = ConditionedUNetModel(
        #     num_channels, num_channels, time_emb_dim, film_hidden_layers
        # )
        self.cnn = CNN2DFiLM(
            (28, 28),
            [32, 32, 64, 64, 128, 256],
            conditioning_dim,
            num_channels,
            film_hidden_layers=film_hidden_layers,
        )

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(self.cnn.conv_channels[-1]),
            SelfAttention(
                self.cnn.conv_channels[-1], self.cnn.conv_channels[-1], num_heads=8
            ),
            nn.ReLU(),
            nn.Conv2d(self.cnn.conv_channels[-1], num_channels, kernel_size=1),
        )

    def forward(self, x, conditioning):
        # Denoise images
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x = self.cnn(x, conditioning)
        return self.output_layer(x)


class ConditionalDiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        conditioning_dim,
        num_time_embeddings,
        time_emb_dim,
        cond_emb_dim=None,
    ):
        super(ConditionalDiffusionModel, self).__init__()
        # self.model = ConditionedUNetModel(
        #     num_channels, num_channels, time_emb_dim, film_hidden_layers
        # )
        self.model = model

        if cond_emb_dim is None:
            cond_emb_dim = conditioning_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(num_time_embeddings),
            nn.Linear(num_time_embeddings, time_emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.conditioning_mlp = nn.Sequential(
            nn.Linear(conditioning_dim, cond_emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_emb_dim, cond_emb_dim),
        )

        self.timesteps = None

    def forward(self, x, t, conditioning):
        time_emb = self.time_mlp(t)
        cond_emb = self.conditioning_mlp(conditioning)
        conditioning = torch.cat([cond_emb, time_emb], dim=-1)
        return self.model(x, conditioning)

    def set_beta_schedule(self, timesteps, beta_schedule):
        self.timesteps = timesteps
        self.register_buffer("betas", beta_schedule(timesteps))
        self.register_buffer("alpha_bars", (1 - self.betas).cumprod(0))
        self.register_buffer("alphas", 1 - self.betas)

        self.register_buffer(
            "alpha_bars_prev",
            torch.nn.functional.pad(self.alpha_bars[:-1], (1, 0), value=1.0),
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            "posterior_vars",
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars),
        )

    def get_beta(self, step):
        return self.betas[step]

    def forward_sample(self, x0, t, eps=None):
        """Sample from a single diffusion path starting at x0 at time t."""
        assert x0.shape[0] == t.shape[0]

        if self.timesteps is None:
            raise ValueError("Must set beta schedule before sampling.")

        # Sample noise
        if eps is None:
            eps = torch.randn_like(x0)

        alpha_bar = self.alpha_bars[t]

        # Sample from model
        for ii in range(len(x0.shape) - 1):
            alpha_bar = alpha_bar.unsqueeze(-1)  # Unsqueeze so shaped the same as x.

        x = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * eps

        return x

    def get_loss(self, x0, conditioning):
        if self.timesteps is None:
            raise ValueError("Must set beta schedule before computing loss.")

        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)

        # Sample random noise
        eps = torch.randn_like(x0)

        # Sample noisy images
        x = self.forward_sample(x0, t, eps=eps)

        noise_pred = self.forward(x, t, conditioning)

        # Compute loss
        loss = nn.functional.smooth_l1_loss(noise_pred, eps)
        # loss = torch.mean(noise_pred**2)

        return loss


    def generate(self, x, n_samples, conditioning, test_step, denoise_true=False):

        x = x.repeat(n_samples, 1)
        conditioning = conditioning.repeat(n_samples, 1)

        x = torch.randn_like(x)
        
        # Iteratively denoise the image
        for tt in reversed(range(test_step)):
            time_tensor = torch.ones(x.shape[0]).long().cuda() * tt
            denoise_pred = self.forward(x, time_tensor, conditioning)

            denoise_scaled = (1 - self.alphas[tt]) / torch.sqrt(1 - self.alpha_bars[tt]) * denoise_pred

            mu_prev = (1 / torch.sqrt(self.alphas[tt])) * (x - denoise_scaled)

            if tt != 0:
                x = mu_prev + torch.sqrt(self.posterior_vars[tt]) * torch.randn_like(x)
                # x = mu_prev + torch.sqrt(model.betas[tt]) * torch.randn_like(x)
            else:
                x = mu_prev
                
        return x