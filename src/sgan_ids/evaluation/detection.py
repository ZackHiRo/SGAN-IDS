"""Downstream intrusion-detection evaluation.

Classifiers are trained on (possibly augmented) *training* data and scored on
the untouched real *test* split. Synthetic rows never appear in any test set.
"""

from __future__ import annotations

import inspect

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier


def make_classifier(name: str, seed: int):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=1, n_jobs=-1, random_state=seed
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, early_stopping=True, random_state=seed
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256, 128),
            batch_size=512,
            max_iter=200,
            early_stopping=True,
            random_state=seed,
        )
    raise ValueError(f"Unknown classifier {name!r}")


def supports_sample_weight(clf) -> bool:
    return "sample_weight" in inspect.signature(clf.fit).parameters


def fit_predict(
    name: str,
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray | None:
    clf = make_classifier(name, seed)
    if sample_weight is not None:
        if not supports_sample_weight(clf):
            return None
        clf.fit(X, y, sample_weight=sample_weight)
    else:
        clf.fit(X, y)
    return clf.predict(X_test)


def detection_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, rare: list[int]
) -> dict[str, float | list[float]]:
    labels = list(range(n_classes))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "rare_macro_f1": float(np.mean(f[rare])) if rare else float("nan"),
        "rare_macro_recall": float(np.mean(r[rare])) if rare else float("nan"),
        "per_class_precision": p.tolist(),
        "per_class_recall": r.tolist(),
        "per_class_f1": f.tolist(),
    }


def bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95 % bootstrap interval of macro-F1 over test rows (sampling uncertainty of the test set)."""
    rng = np.random.default_rng(seed)
    labels = list(range(n_classes))
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(
            f1_score(y_true[idx], y_pred[idx], labels=labels, average="macro", zero_division=0)
        )
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)
