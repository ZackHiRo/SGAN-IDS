from __future__ import annotations

import torch
import numpy as np
from sklearn.metrics import confusion_matrix

__all__ = [
    "gradient_penalty",
    "kl_divergence",
    "js_divergence",
    "confusion_matrix_wrapper",
]


def _score_only(out: torch.Tensor | tuple):
    """Critic may return (score, logits); keep score only."""
    return out[0] if isinstance(out, (tuple, list)) else out


def gradient_penalty(critic, real: torch.Tensor, fake: torch.Tensor, lam: float = 10.0):
    """WGAN-GP gradient penalty (single scalar)."""
    device = real.device
    bsz = real.size(0)
    alpha = torch.rand(bsz, 1, device=device)
    for _ in range(real.dim() - 2):
        alpha = alpha.unsqueeze(-1)
    inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = _score_only(critic(inter))
    grad = torch.autograd.grad(score, inter, torch.ones_like(score), create_graph=True)[0]
    grad = grad.view(bsz, -1)
    return lam * ((grad.norm(2, dim=1) - 1) ** 2).mean()


def kl_divergence(p, q, eps: float = 1e-10):
    """KL(p || q) for discrete distributions given as arrays."""
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p, q, eps: float = 1e-10):
    """Symmetric Jensen–Shannon divergence."""
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def confusion_matrix_wrapper(y_true, y_pred, labels=None, normalize=None):
    """Thin wrapper around sklearn.confusion_matrix."""
    return confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize) 