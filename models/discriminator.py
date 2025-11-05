import torch
from torch import nn
from .attention import SelfAttention1D

# --- Minibatch Discrimination block ---
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims):
        super().__init__()
        print(f'[MBD] in_features={in_features}, out_features={out_features}, kernel_dims={kernel_dims}')
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims) * 0.05)

    def forward(self, x):
        # x: [batch_size, in_features]
        M = x.matmul(self.T.view(x.size(1), -1))  # [batch_size, out_features * kernel_dims]
        M = M.view(-1, self.T.size(1), self.T.size(2))  # [batch_size, out_features, kernel_dims]
        out = []
        for i in range(M.size(0)):
            # Compute L1 distance between sample i and all other samples, per kernel
            diff = torch.abs(M[i] - M)  # [batch_size, out_features, kernel_dims]
            # Sum over kernel_dims, then exp(-sum), then sum over batch (excluding self)
            exp_sum = torch.exp(-diff.sum(2))  # [batch_size, out_features]
            # Exclude self
            o_i = exp_sum.sum(0) - 1
            out.append(o_i)
        out = torch.stack(out)  # [batch_size, out_features]
        return torch.cat([x, out], 1)  # [batch_size, in_features + out_features]

class Discriminator(nn.Module):
    """Simple critic producing Wasserstein score + optional class logits."""

    def __init__(self, in_dim: int, n_classes: int | None = None):
        super().__init__()
        self.embed = nn.Embedding(n_classes, n_classes) if n_classes else None
        net_in_dim = in_dim  # No label embedding concatenation for now
        self.net = nn.Sequential(
            nn.Linear(net_in_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )
        self.attn = SelfAttention1D(64, heads=1)
        # Add minibatch discrimination after attention
        self.mbd = MinibatchDiscrimination(64, 50, 5)
        self.score = nn.Linear(64 + 50, 1)
        self.cls = nn.Linear(64 + 50, n_classes) if n_classes else None

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None, return_features: bool = False):
        # Only concatenate label embedding if you want class conditioning
        # if self.embed is not None and labels is not None:
        #     x = torch.cat([x, self.embed(labels)], 1)
        h = self.attn(self.net(x))
        features = h  # For feature matching
        h = self.mbd(h)
        # Debug print and assert
        #print(f'[Discriminator] h shape after mbd: {h.shape}, expected: ({x.shape[0]}, 114)')
        assert h.shape[1] == 114, f"Shape mismatch after minibatch discrimination: got {h.shape[1]}, expected 114"
        score = self.score(h).squeeze()
        cls_logits = self.cls(h) if self.cls else None
        if return_features:
            return score, cls_logits, features
        return score, cls_logits 