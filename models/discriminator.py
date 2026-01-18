import torch
from torch import nn
from torch.nn.utils import spectral_norm
from .attention import SelfAttention1D

__all__ = ["Discriminator", "MinibatchDiscrimination"]


# --- Minibatch Discrimination block ---
class MinibatchDiscrimination(nn.Module):
    """Minibatch discrimination to prevent mode collapse.
    
    Computes pairwise sample similarities within a batch to encourage diversity.
    """

    def __init__(self, in_features: int, out_features: int, kernel_dims: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims) * 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, in_features]
        batch_size = x.size(0)
        # Efficient vectorized implementation
        M = x.matmul(self.T.view(self.in_features, -1))  # [batch_size, out_features * kernel_dims]
        M = M.view(batch_size, self.out_features, self.kernel_dims)  # [batch_size, out_features, kernel_dims]
        
        # Compute pairwise L1 distances (vectorized)
        M_expanded = M.unsqueeze(0)  # [1, batch_size, out_features, kernel_dims]
        M_transposed = M.unsqueeze(1)  # [batch_size, 1, out_features, kernel_dims]
        diff = torch.abs(M_expanded - M_transposed)  # [batch_size, batch_size, out_features, kernel_dims]
        exp_sum = torch.exp(-diff.sum(dim=3))  # [batch_size, batch_size, out_features]
        
        # Sum over batch dimension, excluding self (diagonal)
        out = exp_sum.sum(dim=1) - 1  # [batch_size, out_features]
        
        return torch.cat([x, out], dim=1)  # [batch_size, in_features + out_features]


class Discriminator(nn.Module):
    """WGAN-GP critic with auxiliary classification head.
    
    Uses LayerNorm (not BatchNorm) for compatibility with gradient penalty.
    Optional spectral normalization for additional stability.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int | None = None,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        mbd_out_features: int = 50,
        mbd_kernel_dims: int = 5,
        use_spectral_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.mbd_out_features = mbd_out_features
        
        # Optional class embedding (for projection discriminator)
        self.embed = nn.Embedding(n_classes, hidden_dims[-1]) if n_classes else None
        
        # Build network layers
        layers = []
        prev_dim = in_dim
        for i, h_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, h_dim)
            if use_spectral_norm:
                linear = spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.LayerNorm(h_dim))  # LayerNorm instead of BatchNorm
            layers.append(nn.LeakyReLU(0.2))
            if dropout > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        self.net = nn.Sequential(*layers)
        self.attn = SelfAttention1D(hidden_dims[-1], heads=1)
        
        # Minibatch discrimination
        self.mbd = MinibatchDiscrimination(hidden_dims[-1], mbd_out_features, mbd_kernel_dims)
        
        # Output heads
        final_dim = hidden_dims[-1] + mbd_out_features
        self.score = nn.Linear(final_dim, 1)
        if use_spectral_norm:
            self.score = spectral_norm(self.score)
        self.cls = nn.Linear(final_dim, n_classes) if n_classes else None

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        h = self.net(x)
        h = self.attn(h)
        features = h  # For feature matching loss
        
        # Optional: projection discriminator (add class embedding to features)
        if self.embed is not None and labels is not None:
            class_emb = self.embed(labels)
            # Use inner product for class conditioning (projection discriminator)
            h = h + class_emb
        
        h = self.mbd(h)
        
        score = self.score(h).squeeze(-1)
        cls_logits = self.cls(h) if self.cls else None
        
        if return_features:
            return score, cls_logits, features
        return score, cls_logits 