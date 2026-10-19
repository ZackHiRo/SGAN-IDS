"""How close are synthetic records to real ones?

All metrics are computed *per class* (a synthetic Worm must look like a real
Worm, not like average traffic) in the encoded [-1, 1] space, against held-out
real data, and then macro-averaged so that rare classes count as much as
frequent ones.

- ``mmd_rbf``: kernel two-sample statistic; also used for checkpoint selection.
- ``prdc``: precision / recall / density / coverage (Naeem et al., 2020).
  Precision and density measure realism, recall and coverage measure diversity
  (a collapsed generator has high precision but low coverage).
- ``c2st_auc``: classifier two-sample test. A classifier is trained to tell
  real from synthetic; 0.5 AUC means it cannot.
- ``marginal_wasserstein``: mean 1-D Wasserstein distance per feature.
- ``correlation_gap``: mean absolute difference of feature correlation matrices.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors


def median_gamma(x: np.ndarray, max_rows: int = 2000, seed: int = 0) -> float:
    """RBF bandwidth from the median heuristic, gamma = 1 / median(||xi - xj||^2)."""
    rng = np.random.default_rng(seed)
    if len(x) > max_rows:
        x = x[rng.choice(len(x), max_rows, replace=False)]
    t = torch.as_tensor(x, dtype=torch.float64)
    d = torch.cdist(t, t).pow(2)
    med = d[d > 0].median().item() if (d > 0).any() else 1.0
    return 1.0 / med


def mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Unbiased estimate of squared MMD with an RBF kernel."""
    a = torch.as_tensor(x, dtype=torch.float64)
    b = torch.as_tensor(y, dtype=torch.float64)
    n, m = len(a), len(b)
    kxx = torch.exp(-gamma * torch.cdist(a, a).pow(2))
    kyy = torch.exp(-gamma * torch.cdist(b, b).pow(2))
    kxy = torch.exp(-gamma * torch.cdist(a, b).pow(2))
    value = (kxx.sum() - n) / (n * (n - 1)) + (kyy.sum() - m) / (m * (m - 1)) - 2 * kxy.mean()
    return float(value)


def prdc(real: np.ndarray, fake: np.ndarray, k: int = 5) -> dict[str, float]:
    """Precision, recall, density and coverage of Naeem et al. (2020)."""
    k = min(k, len(real) - 1, len(fake) - 1)
    real_radius = NearestNeighbors(n_neighbors=k + 1).fit(real).kneighbors(real)[0][:, -1]
    fake_radius = NearestNeighbors(n_neighbors=k + 1).fit(fake).kneighbors(fake)[0][:, -1]
    d_real_fake = torch.cdist(
        torch.as_tensor(real), torch.as_tensor(fake)
    ).numpy()  # (n_real, n_fake)

    inside_real = d_real_fake < real_radius[:, None]  # fake j inside ball of real i
    precision = inside_real.any(axis=0).mean()
    recall = (d_real_fake < fake_radius[None, :]).any(axis=1).mean()
    density = inside_real.sum(axis=0).mean() / k
    coverage = (d_real_fake.min(axis=1) < real_radius).mean()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "density": float(density),
        "coverage": float(coverage),
    }


def c2st_auc(real: np.ndarray, fake: np.ndarray, seed: int = 0) -> float:
    """Cross-validated AUC of a random forest separating real from synthetic rows."""
    n = min(len(real), len(fake))
    rng = np.random.default_rng(seed)
    real = real[rng.choice(len(real), n, replace=False)]
    fake = fake[rng.choice(len(fake), n, replace=False)]
    x = np.vstack([real, fake])
    y = np.r_[np.zeros(n), np.ones(n)]
    folds = min(5, n)
    if folds < 2:
        return float("nan")
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    proba = cross_val_predict(
        clf,
        x,
        y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
        method="predict_proba",
    )[:, 1]
    return float(roc_auc_score(y, proba))


def marginal_wasserstein(real: np.ndarray, fake: np.ndarray) -> float:
    return float(
        np.mean([wasserstein_distance(real[:, j], fake[:, j]) for j in range(real.shape[1])])
    )


def correlation_gap(real: np.ndarray, fake: np.ndarray) -> float:
    with np.errstate(invalid="ignore", divide="ignore"):
        cr = np.nan_to_num(np.corrcoef(real, rowvar=False))
        cf = np.nan_to_num(np.corrcoef(fake, rowvar=False))
    iu = np.triu_indices_from(cr, k=1)
    return float(np.abs(cr[iu] - cf[iu]).mean())


def class_fidelity(
    real: np.ndarray, fake: np.ndarray, k: int = 5, max_samples: int = 2000, seed: int = 0
) -> dict[str, float]:
    """All fidelity metrics for one class."""
    rng = np.random.default_rng(seed)
    if len(real) > max_samples:
        real = real[rng.choice(len(real), max_samples, replace=False)]
    if len(fake) > max_samples:
        fake = fake[rng.choice(len(fake), max_samples, replace=False)]
    out = {
        "n_real": len(real),
        "mmd": mmd_rbf(real, fake, median_gamma(real, seed=seed)),
        "c2st_auc": c2st_auc(real, fake, seed=seed),
        "wasserstein": marginal_wasserstein(real, fake),
        "correlation_gap": correlation_gap(real, fake),
    }
    out.update(prdc(real, fake, k=k))
    return out
