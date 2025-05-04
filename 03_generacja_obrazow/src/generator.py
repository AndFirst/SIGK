from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalDiffusionModel(nn.Module):
    def __init__(
        self,
        num_labels: int = 8,
        embed_dim: int = 8,
        time_dim: int = 128,
        num_timesteps: int = 500,
        img_channels: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_channels = img_channels
        self.time_dim = time_dim
        self.num_timesteps = num_timesteps

        # Embeddings
        self.label_embedding = nn.Embedding(num_labels, embed_dim)
        self.time_emb_proj = nn.Sequential(
            nn.Linear(time_dim, 512), nn.SiLU(), nn.Linear(512, 512)
        )

        # Encoder
        self.encoder = nn.ModuleList(
            [
                ResidualBlock(img_channels + embed_dim, 64),
                ResidualBlock(64, 128),
                ResidualBlock(128, 256),
                ResidualBlock(256, 512),
            ]
        )
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(ResidualBlock(512, 512), SelfAttention(512))

        # Decoder
        self.decoder = nn.ModuleList(
            [
                nn.ConvTranspose2d(512, 256, 2, stride=2),
                ResidualBlock(512 + 256, 256),
                nn.ConvTranspose2d(256, 128, 2, stride=2),
                ResidualBlock(256 + 128, 128),
                nn.ConvTranspose2d(128, 64, 2, stride=2),
                ResidualBlock(128 + 64, 64),
                nn.ConvTranspose2d(64, 32, 2, stride=2),
                ResidualBlock(64 + 32, 32),
            ]
        )
        self.out = nn.Conv2d(32, img_channels, 3, padding=1)

        # Noise scheduler
        self.register_buffer("betas", torch.linspace(0.0001, 0.02, num_timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_cumprod", torch.cumprod(self.alphas, dim=0))

    def get_sinusoidal_embedding(
        self, timesteps: torch.Tensor, dim: int = 128
    ) -> torch.Tensor:
        half_dim = dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward_diffusion(
        self, x0: torch.Tensor, t: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod = self.alpha_cumprod[t].sqrt()[:, None, None, None]
        sqrt_one_minus_alpha_cumprod = (1.0 - self.alpha_cumprod[t]).sqrt()[
            :, None, None, None
        ]
        xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
        return xt, noise

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # Label embedding
        y_embed = self.label_embedding(y)[:, :, None, None].expand(
            -1, -1, x.shape[2], x.shape[3]
        )
        x_cat = torch.cat([x, y_embed], dim=1)

        # Time embedding
        t_emb = self.time_emb_proj(self.get_sinusoidal_embedding(t, self.time_dim))
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)

        # Encoder
        skips = []
        h = x_cat
        for enc in self.encoder:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)

        # Bottleneck
        h = self.bottleneck(h) + t_emb

        # Decoder
        for i in range(0, len(self.decoder), 2):
            h = self.decoder[i](h)
            skip = skips[-(i // 2 + 1)]
            h = self.decoder[i + 1](torch.cat([h, skip], dim=1))

        return self.out(h)

    def sample(
        self,
        label: int,
        num_samples: int,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
    ) -> torch.Tensor:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device).eval()
        samples = []

        for start_idx in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - start_idx)
            x_t = torch.randn(
                current_batch_size, self.img_channels, 64, 64, device=device
            )
            y_tensor = torch.full(
                (current_batch_size,), label, dtype=torch.long, device=device
            )

            with torch.no_grad():
                for t in reversed(range(self.num_timesteps)):
                    t_tensor = torch.full(
                        (current_batch_size,), t, dtype=torch.long, device=device
                    )
                    pred_noise = self.forward(x_t, t_tensor, y_tensor)
                    alpha = self.alphas[t]
                    alpha_cumprod = self.alpha_cumprod[t]
                    beta = self.betas[t]
                    noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
                    x_t = (
                        x_t - (1 - alpha) / (1 - alpha_cumprod).sqrt() * pred_noise
                    ) / alpha.sqrt() + beta.sqrt() * noise

            samples.append(torch.clamp(x_t, -1, 1))

        self.train()
        return torch.cat(samples, dim=0)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, model_config: dict
    ) -> "ConditionalDiffusionModel":
        model = cls(**model_config)
        model.load_state_dict(torch.load(checkpoint_path))
        return model


class SelfAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.qkv = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.proj = nn.Conv1d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        qkv = self.qkv(x_flat)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        attn_scores = torch.bmm(q.transpose(1, 2), k) / math.sqrt(C)
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.bmm(v, attn_weights.transpose(1, 2)).view(B, C, H, W)
        return self.proj(out.view(B, C, -1)).view(B, C, H, W) + x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))
