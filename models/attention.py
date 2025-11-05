import torch
from torch import nn

__all__ = ["SelfAttention1D"]


class SelfAttention1D(nn.Module):
    """Lightweight self-attention for 1-D feature tensors (B, F)."""

    def __init__(self, feat_dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert feat_dim % heads == 0, "feat_dim must be divisible by heads"
        self.h, self.d = heads, feat_dim // heads
        self.scale = self.d ** -0.5
        self.proj_qkv = nn.Linear(feat_dim, feat_dim * 3, bias=False)
        self.proj_out = nn.Linear(feat_dim, feat_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, F = x.shape
        qkv = self.proj_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, self.h, self.d) for t in qkv]
        attn = (q * k).sum(-1) * self.scale  # (B, H)
        attn = self.drop(attn.softmax(dim=-1))
        ctx = (attn.unsqueeze(-1) * v).reshape(B, F)
        return self.proj_out(ctx) + x  # residual 