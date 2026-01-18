"""Attention mechanisms for tabular/1D data.

Provides self-attention variants suitable for feature vectors:
- SelfAttention1D: Lightweight attention for flat feature vectors
- FeatureGroupAttention: Attention across feature groups
"""

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["SelfAttention1D", "FeatureGroupAttention"]


class SelfAttention1D(nn.Module):
    """Lightweight self-attention for 1-D feature tensors (B, F).
    
    For flat feature vectors, this implements a gated attention mechanism
    that learns to weight different features. Uses a residual connection.
    
    Args:
        feat_dim: Feature dimension
        heads: Number of attention heads (feat_dim must be divisible by heads)
        dropout: Dropout rate
    """

    def __init__(self, feat_dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        # Ensure divisibility, fall back to 1 head if needed
        if feat_dim % heads != 0:
            heads = 1
        
        self.feat_dim = feat_dim
        self.heads = heads
        self.head_dim = feat_dim // heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (B, feat_dim)
            
        Returns:
            Output tensor of shape (B, feat_dim)
        """
        batch_size, n_features = x.shape
        
        # Pre-norm
        x_norm = self.norm(x)
        
        # Project to Q, K, V
        q = self.q_proj(x_norm).view(batch_size, self.heads, self.head_dim)
        k = self.k_proj(x_norm).view(batch_size, self.heads, self.head_dim)
        v = self.v_proj(x_norm).view(batch_size, self.heads, self.head_dim)
        
        # Compute attention scores (per head, per sample)
        # For flat features, we compute attention within each head
        attn_scores = (q * k).sum(dim=-1) * self.scale  # (B, H)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # Weight each head's values by its attention score
        context = (attn_weights.unsqueeze(-1) * v).view(batch_size, n_features)
        
        # Output projection + residual
        out = self.out_proj(context)
        return x + out


class FeatureGroupAttention(nn.Module):
    """Attention mechanism that operates on groups of features.
    
    Useful when features can be logically grouped (e.g., by source, type).
    Treats each group as a "token" and applies standard transformer attention.
    
    Args:
        feat_dim: Total feature dimension
        n_groups: Number of feature groups
        heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        feat_dim: int,
        n_groups: int = 8,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Compute group size (pad if necessary)
        self.n_groups = n_groups
        self.group_size = (feat_dim + n_groups - 1) // n_groups
        self.padded_dim = self.group_size * n_groups
        self.feat_dim = feat_dim
        
        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=self.group_size,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(self.group_size, self.group_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.group_size * 4, self.group_size),
            nn.Dropout(dropout),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(self.group_size)
        self.norm2 = nn.LayerNorm(self.group_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, feat_dim)
            
        Returns:
            Output tensor of shape (B, feat_dim)
        """
        batch_size, n_features = x.shape
        
        # Pad if necessary
        if n_features < self.padded_dim:
            x = F.pad(x, (0, self.padded_dim - n_features))
        
        # Reshape to (B, n_groups, group_size)
        x = x.view(batch_size, self.n_groups, self.group_size)
        
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        # Reshape back and remove padding
        x = x.view(batch_size, -1)[:, :self.feat_dim]
        
        return x 