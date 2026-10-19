"""How separable is each rare class *at all* with these features?

If a test row's exact feature vector also occurs in the training data under a
different label, no classifier can be sure of it, whatever augmentation is
used. The share of such rows bounds what any rebalancing method can achieve.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _keys(x: np.ndarray) -> list[bytes]:
    x = np.ascontiguousarray(x)
    return [row.tobytes() for row in x]


def label_overlap(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[int],
) -> list[dict]:
    labels_by_row: dict[bytes, set[int]] = defaultdict(set)
    for key, label in zip(_keys(X_train), y_train):
        labels_by_row[key].add(int(label))

    rows = []
    for c in classes:
        keys = _keys(X_test[y_test == c])
        seen = [labels_by_row.get(k) for k in keys]
        n = max(len(keys), 1)
        conflict = sum(1 for s in seen if s and (s - {c}))
        same_only = sum(1 for s in seen if s == {c})
        train_keys = _keys(X_train[y_train == c])
        train_conflict = sum(1 for k in train_keys if labels_by_row[k] - {c})
        rows.append(
            {
                "class": c,
                "n_test": len(keys),
                "test_seen_with_other_label": conflict / n,
                "test_seen_with_same_label_only": same_only / n,
                "test_unseen": 1 - (conflict + same_only) / n,
                "train_shared_with_other_label": train_conflict / max(len(train_keys), 1),
            }
        )
    return rows
