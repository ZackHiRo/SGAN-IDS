"""Generator and critic for a class-conditional WGAN-GP on mixed-type tabular data."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sgan_ids.data.transform import Span


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.LeakyReLU(0.2), nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Generator(nn.Module):
    """Maps (noise, class) to an encoded record.

    The output layer is split into the transformer's spans: numeric columns go
    through ``tanh`` (data lives in [-1, 1]) and each categorical block through
    a Gumbel-softmax, which keeps gradients flowing while producing near-one-hot
    vectors (as in CTGAN, Xu et al. 2019). Without it, the critic can separate
    real from fake simply because fake one-hots are never exactly 0/1.
    """

    def __init__(
        self,
        latent_dim: int,
        n_classes: int,
        spans: list[Span],
        hidden: list[int],
        class_embed_dim: int = 32,
        gumbel_tau: float = 0.2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.spans = list(spans)
        self.gumbel_tau = gumbel_tau
        out_dim = self.spans[-1].end

        self.embed = nn.Embedding(n_classes, class_embed_dim)
        layers: list[nn.Module] = [
            nn.Linear(latent_dim + class_embed_dim, hidden[0]),
            nn.LeakyReLU(0.2),
        ]
        for i in range(1, len(hidden)):
            if hidden[i] == hidden[i - 1]:
                layers.append(ResidualBlock(hidden[i]))
            else:
                layers += [nn.Linear(hidden[i - 1], hidden[i]), nn.LeakyReLU(0.2)]
        layers += [nn.LayerNorm(hidden[-1]), nn.Linear(hidden[-1], out_dim)]
        self.net = nn.Sequential(*layers)

        numeric = [i for s in self.spans if s.kind == "numeric" for i in range(s.start, s.end)]
        self.register_buffer("numeric_idx", torch.tensor(numeric, dtype=torch.long))
        self.categorical_spans = [(s.start, s.end) for s in self.spans if s.kind == "categorical"]

    def forward(self, z: torch.Tensor, y: torch.Tensor, hard: bool = False) -> torch.Tensor:
        raw = self.net(torch.cat([z, self.embed(y)], dim=1))
        out = torch.empty_like(raw)
        out[:, self.numeric_idx] = torch.tanh(raw[:, self.numeric_idx])
        for start, end in self.categorical_spans:
            out[:, start:end] = F.gumbel_softmax(raw[:, start:end], tau=self.gumbel_tau, hard=hard)
        return out

    @torch.no_grad()
    def sample(self, y: torch.Tensor) -> torch.Tensor:
        z = torch.randn(len(y), self.latent_dim, device=y.device)
        return self(z, y, hard=True)


class Critic(nn.Module):
    """WGAN critic with class conditioning.

    ``conditioning="projection"`` follows Miyato & Koyama (2018):
    ``D(x, y) = w^T phi(x) + embed(y)^T phi(x)``. The class enters through an
    inner product with the features instead of being concatenated to the input,
    which forces the critic to judge "is this a realistic *member of class y*".
    ``"concat"`` appends a one-hot label to the input (kept as an ablation).

    No batch normalisation: the gradient penalty is defined per sample, and
    batch statistics would couple samples together (Gulrajani et al., 2017).
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        hidden: list[int],
        conditioning: str = "projection",
        dropout: float = 0.0,
    ):
        super().__init__()
        if conditioning not in {"projection", "concat"}:
            raise ValueError(f"Unknown conditioning {conditioning!r}")
        self.conditioning = conditioning
        self.n_classes = n_classes
        dim = in_dim + (n_classes if conditioning == "concat" else 0)
        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(dim, h), nn.LeakyReLU(0.2)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = h
        self.features = nn.Sequential(*layers)
        self.linear = nn.Linear(dim, 1)
        if conditioning == "projection":
            self.embed = nn.Embedding(n_classes, dim)
            nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.conditioning == "concat":
            x = torch.cat([x, F.one_hot(y, self.n_classes).to(x.dtype)], dim=1)
        h = self.features(x)
        score = self.linear(h).squeeze(1)
        if self.conditioning == "projection":
            score = score + (self.embed(y) * h).sum(dim=1)
        return score


def gradient_penalty(
    critic: Critic, real: torch.Tensor, fake: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """WGAN-GP penalty (Gulrajani et al., 2017) on real/fake interpolates of the same class."""
    alpha = torch.rand(real.size(0), 1, device=real.device)
    inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = critic(inter, y)
    (grad,) = torch.autograd.grad(score.sum(), inter, create_graph=True)
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
