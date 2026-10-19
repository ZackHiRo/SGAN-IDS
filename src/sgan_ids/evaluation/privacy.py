"""Memorisation check: distance to closest record (DCR).

For each synthetic row of class c we measure the distance to the closest
*training* row of class c, and compare it with the same distance measured for
real *held-out* rows of class c. A generator that generalises produces rows
about as far from the training set as genuinely new traffic is (ratio ~ 1).
A ratio near 0 means it is copying training rows; random oversampling is
exactly 0 by construction.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def dcr_ratio(
    train: np.ndarray, holdout: np.ndarray, synth: np.ndarray, max_rows: int = 2000, seed: int = 0
) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    def sub(x: np.ndarray) -> np.ndarray:
        return x[rng.choice(len(x), max_rows, replace=False)] if len(x) > max_rows else x

    nn = NearestNeighbors(n_neighbors=1).fit(train)
    d_synth = nn.kneighbors(sub(synth))[0][:, 0]
    d_hold = nn.kneighbors(sub(holdout))[0][:, 0]
    return {
        "dcr_synth_median": float(np.median(d_synth)),
        "dcr_holdout_median": float(np.median(d_hold)),
        "dcr_ratio": float(np.median(d_synth) / max(np.median(d_hold), 1e-12)),
        "exact_copy_rate": float(np.mean(d_synth < 1e-6)),
    }
