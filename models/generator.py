import torch
from torch import nn
from .attention import SelfAttention1D


class Generator(nn.Module):
    """Fully-connected generator with optional class conditioning.

    Supports configurable class embedding dimension to match checkpoints trained
    with an embedding size different from the number of classes.
    """

    def __init__(self, latent_dim: int, out_dim: int, n_classes: int | None = None, class_embed_dim: int | None = None):
        super().__init__()
        if n_classes:
            embed_dim = class_embed_dim if class_embed_dim is not None else n_classes
            self.embed = nn.Embedding(n_classes, embed_dim)
            cond_dim = embed_dim
        else:
            self.embed = None
            cond_dim = 0
        in_dim = latent_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, out_dim),
            nn.Tanh(),
        )
        self.attn = SelfAttention1D(out_dim, heads=1)

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None):
        if self.embed is not None and labels is not None:
            z = torch.cat([z, self.embed(labels)], 1)
        x = self.net(z)
        return self.attn(x) 