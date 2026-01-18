"""Metrics for GAN training and evaluation.

Includes:
- Gradient penalty for WGAN-GP
- Distribution divergence metrics (KL, JS)
- Maximum Mean Discrepancy (MMD)
- Precision/Recall for generative models
- Per-class quality metrics
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.stats import wasserstein_distance

__all__ = [
    "gradient_penalty",
    "kl_divergence",
    "js_divergence",
    "mmd_rbf",
    "mmd_linear",
    "compute_precision_recall",
    "compute_coverage_density",
    "per_class_quality_metrics",
    "wasserstein_per_feature",
    "feature_correlation_distance",
    "confusion_matrix_wrapper",
]


# =============================================================================
# GAN Training Metrics
# =============================================================================

def _score_only(out: torch.Tensor | tuple) -> torch.Tensor:
    """Extract critic score from output tuple."""
    return out[0] if isinstance(out, (tuple, list)) else out


def gradient_penalty(
    critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    lam: float = 10.0,
) -> torch.Tensor:
    """WGAN-GP gradient penalty.
    
    Enforces Lipschitz constraint by penalizing gradient norm deviation from 1.
    
    Args:
        critic: Discriminator/critic network
        real: Real samples [batch_size, features]
        fake: Generated samples [batch_size, features]
        lam: Penalty coefficient (default 10.0)
        
    Returns:
        Gradient penalty loss (scalar tensor)
    """
    device = real.device
    bsz = real.size(0)
    
    # Random interpolation coefficient
    alpha = torch.rand(bsz, 1, device=device)
    
    # Interpolate between real and fake
    inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    
    # Get critic scores
    score = _score_only(critic(inter))
    
    # Compute gradients
    grad = torch.autograd.grad(
        outputs=score,
        inputs=inter,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    grad = grad.view(bsz, -1)
    grad_norm = grad.norm(2, dim=1)
    
    return lam * ((grad_norm - 1) ** 2).mean()


# =============================================================================
# Distribution Divergence Metrics
# =============================================================================

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Kullback-Leibler divergence KL(p || q).
    
    Args:
        p: First distribution (reference)
        q: Second distribution (approximation)
        eps: Small constant for numerical stability
        
    Returns:
        KL divergence value
    """
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Jensen-Shannon divergence (symmetric).
    
    Args:
        p: First distribution
        q: Second distribution
        eps: Small constant for numerical stability
        
    Returns:
        JS divergence value (in [0, ln(2)])
    """
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m, eps=0) + kl_divergence(q, m, eps=0))


# =============================================================================
# Maximum Mean Discrepancy (MMD)
# =============================================================================

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """Compute RBF kernel matrix."""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    distances = XX + YY.T - 2 * X @ Y.T
    return np.exp(-gamma * distances)


def mmd_rbf(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: Optional[float] = None,
) -> float:
    """Maximum Mean Discrepancy with RBF kernel.
    
    Measures the distance between two distributions in a reproducing kernel
    Hilbert space. Lower values indicate more similar distributions.
    
    Args:
        X: Samples from first distribution [n_samples, n_features]
        Y: Samples from second distribution [m_samples, n_features]
        gamma: RBF kernel bandwidth (1/2σ²). If None, uses median heuristic.
        
    Returns:
        MMD² value (unbiased estimate)
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    
    n, m = len(X), len(Y)
    
    if gamma is None:
        # Median heuristic for bandwidth selection
        combined = np.vstack([X[:min(1000, n)], Y[:min(1000, m)]])
        pairwise_sq = np.sum((combined[:, None] - combined[None, :]) ** 2, axis=2)
        median_sq = np.median(pairwise_sq[pairwise_sq > 0])
        gamma = 1.0 / (2 * median_sq) if median_sq > 0 else 1.0
    
    K_XX = _rbf_kernel(X, X, gamma)
    K_YY = _rbf_kernel(Y, Y, gamma)
    K_XY = _rbf_kernel(X, Y, gamma)
    
    # Unbiased MMD² estimate
    mmd_sq = (
        (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1))
        + (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))
        - 2 * K_XY.mean()
    )
    
    return float(max(0, mmd_sq))  # Clamp to avoid numerical issues


def mmd_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """Maximum Mean Discrepancy with linear kernel.
    
    Faster than RBF but less expressive. Essentially compares means.
    
    Args:
        X: Samples from first distribution
        Y: Samples from second distribution
        
    Returns:
        MMD² value
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    
    mean_X = X.mean(axis=0)
    mean_Y = Y.mean(axis=0)
    
    return float(np.sum((mean_X - mean_Y) ** 2))


# =============================================================================
# Precision/Recall and Coverage/Density Metrics
# =============================================================================

def compute_precision_recall(
    real: np.ndarray,
    synth: np.ndarray,
    k: int = 5,
) -> Tuple[float, float]:
    """Compute precision and recall for generative models.
    
    Based on "Improved Precision and Recall Metric for Assessing Generative Models"
    (Kynkäänniemi et al., 2019).
    
    - Precision: fraction of synthetic samples that are close to real samples
    - Recall: fraction of real samples that are close to synthetic samples
    
    Args:
        real: Real samples [n, d]
        synth: Synthetic samples [m, d]
        k: Number of nearest neighbors
        
    Returns:
        (precision, recall) tuple
    """
    real = np.asarray(real, dtype=np.float64)
    synth = np.asarray(synth, dtype=np.float64)
    
    # Build KNN models
    nn_real = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(real)
    nn_synth = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(synth)
    
    # Get k-th nearest neighbor distances (radius of hypersphere)
    real_radii = nn_real.kneighbors(real)[0][:, -1]
    synth_radii = nn_synth.kneighbors(synth)[0][:, -1]
    
    # Precision: how many synth samples fall within real manifold
    synth_to_real_dist = nn_real.kneighbors(synth)[0][:, 0]
    precision = np.mean(synth_to_real_dist <= real_radii[nn_real.kneighbors(synth, n_neighbors=1)[1][:, 0]])
    
    # Recall: how many real samples fall within synth manifold  
    real_to_synth_dist = nn_synth.kneighbors(real)[0][:, 0]
    recall = np.mean(real_to_synth_dist <= synth_radii[nn_synth.kneighbors(real, n_neighbors=1)[1][:, 0]])
    
    return float(precision), float(recall)


def compute_coverage_density(
    real: np.ndarray,
    synth: np.ndarray,
    k: int = 5,
) -> Tuple[float, float]:
    """Compute coverage and density metrics.
    
    Based on "Reliable Fidelity and Diversity Metrics for Generative Models"
    (Naeem et al., 2020).
    
    - Coverage: fraction of real samples whose nearest neighbor is a synthetic sample
    - Density: average number of synthetic samples in the neighborhood of real samples
    
    Args:
        real: Real samples [n, d]
        synth: Synthetic samples [m, d]
        k: Number of nearest neighbors
        
    Returns:
        (coverage, density) tuple
    """
    real = np.asarray(real, dtype=np.float64)
    synth = np.asarray(synth, dtype=np.float64)
    
    n_real = len(real)
    n_synth = len(synth)
    
    # Fit KNN on real samples
    nn_real = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(real)
    
    # Get distances and radii
    real_distances, _ = nn_real.kneighbors(real)
    radii = real_distances[:, -1]  # k-th neighbor distance
    
    # For each synthetic sample, find nearest real sample
    synth_to_real_dist, synth_to_real_idx = nn_real.kneighbors(synth, n_neighbors=1)
    synth_to_real_dist = synth_to_real_dist[:, 0]
    synth_to_real_idx = synth_to_real_idx[:, 0]
    
    # Coverage: count real samples that have at least one synth sample in their ball
    real_covered = np.zeros(n_real, dtype=bool)
    for i, (dist, idx) in enumerate(zip(synth_to_real_dist, synth_to_real_idx)):
        if dist <= radii[idx]:
            real_covered[idx] = True
    coverage = real_covered.mean()
    
    # Density: average number of synth samples per real ball
    synth_in_ball = np.zeros(n_real)
    for i, (dist, idx) in enumerate(zip(synth_to_real_dist, synth_to_real_idx)):
        if dist <= radii[idx]:
            synth_in_ball[idx] += 1
    density = synth_in_ball.mean() / k  # Normalize by k
    
    return float(coverage), float(density)


# =============================================================================
# Per-Feature and Per-Class Metrics
# =============================================================================

def wasserstein_per_feature(
    real: np.ndarray,
    synth: np.ndarray,
) -> np.ndarray:
    """Compute Wasserstein distance for each feature separately.
    
    Args:
        real: Real samples [n, d]
        synth: Synthetic samples [m, d]
        
    Returns:
        Array of Wasserstein distances [d]
    """
    real = np.asarray(real, dtype=np.float64)
    synth = np.asarray(synth, dtype=np.float64)
    
    n_features = real.shape[1]
    distances = np.zeros(n_features)
    
    for i in range(n_features):
        distances[i] = wasserstein_distance(real[:, i], synth[:, i])
    
    return distances


def feature_correlation_distance(
    real: np.ndarray,
    synth: np.ndarray,
) -> float:
    """Compute distance between correlation matrices.
    
    Measures how well the synthetic data preserves feature correlations.
    
    Args:
        real: Real samples [n, d]
        synth: Synthetic samples [m, d]
        
    Returns:
        Frobenius norm of correlation matrix difference
    """
    real = np.asarray(real, dtype=np.float64)
    synth = np.asarray(synth, dtype=np.float64)
    
    corr_real = np.corrcoef(real.T)
    corr_synth = np.corrcoef(synth.T)
    
    # Handle NaN (constant features)
    corr_real = np.nan_to_num(corr_real, nan=0.0)
    corr_synth = np.nan_to_num(corr_synth, nan=0.0)
    
    return float(np.linalg.norm(corr_real - corr_synth, 'fro'))


def per_class_quality_metrics(
    real: np.ndarray,
    synth: np.ndarray,
    y_real: np.ndarray,
    y_synth: np.ndarray,
    n_classes: Optional[int] = None,
) -> Dict[int, Dict[str, float]]:
    """Compute quality metrics for each class separately.
    
    Args:
        real: Real samples [n, d]
        synth: Synthetic samples [m, d]
        y_real: Real labels [n]
        y_synth: Synthetic labels [m]
        n_classes: Number of classes (inferred if None)
        
    Returns:
        Dict mapping class_id -> {metric_name: value}
    """
    real = np.asarray(real, dtype=np.float64)
    synth = np.asarray(synth, dtype=np.float64)
    y_real = np.asarray(y_real)
    y_synth = np.asarray(y_synth)
    
    if n_classes is None:
        n_classes = max(y_real.max(), y_synth.max()) + 1
    
    results = {}
    
    for cls in range(n_classes):
        real_cls = real[y_real == cls]
        synth_cls = synth[y_synth == cls]
        
        if len(real_cls) < 10 or len(synth_cls) < 10:
            # Skip classes with too few samples
            results[cls] = {
                "n_real": len(real_cls),
                "n_synth": len(synth_cls),
                "mmd": np.nan,
                "precision": np.nan,
                "recall": np.nan,
            }
            continue
        
        # Subsample for efficiency
        max_samples = 2000
        if len(real_cls) > max_samples:
            idx = np.random.choice(len(real_cls), max_samples, replace=False)
            real_cls = real_cls[idx]
        if len(synth_cls) > max_samples:
            idx = np.random.choice(len(synth_cls), max_samples, replace=False)
            synth_cls = synth_cls[idx]
        
        try:
            mmd = mmd_rbf(real_cls, synth_cls)
            precision, recall = compute_precision_recall(real_cls, synth_cls, k=3)
        except Exception:
            mmd, precision, recall = np.nan, np.nan, np.nan
        
        results[cls] = {
            "n_real": len(real[y_real == cls]),
            "n_synth": len(synth[y_synth == cls]),
            "mmd": mmd,
            "precision": precision,
            "recall": recall,
        }
    
    return results


# =============================================================================
# Utilities
# =============================================================================

def confusion_matrix_wrapper(y_true, y_pred, labels=None, normalize=None):
    """Thin wrapper around sklearn.confusion_matrix."""
    return confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize) 