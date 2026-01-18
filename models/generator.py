import torch
from torch import nn
from .attention import SelfAttention1D

__all__ = ["Generator", "ResidualBlock"]


class ResidualBlock(nn.Module):
    """Residual MLP block with LayerNorm and dropout."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class Generator(nn.Module):
    """Class-conditional generator with residual blocks and LayerNorm.

    Architecture improvements over vanilla MLP:
    - Residual connections for better gradient flow
    - LayerNorm instead of BatchNorm (works better with small batches)
    - Configurable depth and width
    - Optional self-attention
    """

    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        n_classes: int | None = None,
        class_embed_dim: int | None = None,
        hidden_dims: tuple[int, ...] = (256, 512, 512, 256),
        n_residual_blocks: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.n_classes = n_classes
        
        # Class embedding
        if n_classes:
            embed_dim = class_embed_dim if class_embed_dim is not None else min(n_classes, 64)
            self.embed = nn.Embedding(n_classes, embed_dim)
            cond_dim = embed_dim
        else:
            self.embed = None
            cond_dim = 0
        
        in_dim = latent_dim + cond_dim
        
        # Build the main network
        layers = []
        prev_dim = in_dim
        
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.LeakyReLU(0.2))
            if dropout > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        self.net = nn.Sequential(*layers)
        
        # Residual blocks at the bottleneck
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dims[-1], dropout) for _ in range(n_residual_blocks)]
        )
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], out_dim),
            nn.Tanh(),
        )
        
        # Optional attention
        self.use_attention = use_attention
        if use_attention:
            self.attn = SelfAttention1D(out_dim, heads=1)

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        # Concatenate noise with class embedding
        if self.embed is not None and labels is not None:
            z = torch.cat([z, self.embed(labels)], dim=1)
        
        # Forward through network
        h = self.net(z)
        h = self.residual_blocks(h)
        x = self.out_proj(h)
        
        # Optional attention
        if self.use_attention:
            x = self.attn(x)
        
        return x 