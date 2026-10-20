"""Rebalancing strategies compared in the experiments.

Every strategy receives the same encoded training set and the same
augmentation plan (how many rows to add per class), so the only thing that
differs between them is *where the extra rows come from*:

=============  =================================================================
none           real data only (reference point)
class_weight   real data only, loss re-weighted inversely to class frequency
ros            random oversampling: duplicate existing rare rows
smote          SMOTE: interpolate between a rare row and one of its k neighbours
ctgan          CTGAN (Xu et al., 2019): off-the-shelf conditional tabular GAN
gan            this project's conditional WGAN-GP
=============  =================================================================
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.utils.class_weight import compute_sample_weight

from sgan_ids.data.prepare import PreparedData
from sgan_ids.models.networks import Generator

GENERATIVE = ("ros", "smote", "ctgan", "gan")


@dataclass
class Augmented:
    X: np.ndarray
    y: np.ndarray
    sample_weight: np.ndarray | None = None
    X_synth: np.ndarray | None = None
    y_synth: np.ndarray | None = None


def augmentation_plan(
    y: np.ndarray, n_classes: int, target: int, classes: list[int] | None = None
) -> dict[int, int]:
    """Rows to add per class so that each class in ``classes`` (default: all) has ``target`` rows."""
    counts = np.bincount(y, minlength=n_classes)
    allowed = range(n_classes) if classes is None else classes
    return {c: int(target - counts[c]) for c in sorted(allowed) if 0 < counts[c] < target}


def resolve_classes(names: list[str], classes: list[str]) -> list[int] | None:
    """Map configured class names to indices (None when no list is configured)."""
    if not names:
        return None
    unknown = sorted(set(names) - set(classes))
    if unknown:
        raise ValueError(f"augment.classes not found in the data: {unknown}; available: {classes}")
    return sorted(classes.index(n) for n in names)


def combine(data: PreparedData, X_synth: np.ndarray, y_synth: np.ndarray) -> Augmented:
    return Augmented(
        np.vstack([data.X_train, X_synth]).astype(np.float32),
        np.r_[data.y_train, y_synth],
        None,
        X_synth,
        y_synth,
    )


def random_oversample(
    data: PreparedData, plan: dict[int, int], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.concatenate(
        [rng.choice(np.flatnonzero(data.y_train == c), n) for c, n in plan.items()]
    )
    return data.X_train[idx], data.y_train[idx]


def smote(data: PreparedData, plan: dict[int, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    from imblearn.over_sampling import SMOTE

    counts = np.bincount(data.y_train, minlength=data.n_classes)
    strategy = {c: int(counts[c] + n) for c, n in plan.items()}
    k = int(min(5, min(counts[c] for c in plan) - 1))
    X_res, y_res = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed).fit_resample(
        data.X_train, data.y_train
    )
    X_new, y_new = X_res[len(data.X_train) :], y_res[len(data.X_train) :]
    # Interpolated one-hots are not valid categories; snap them like GAN output.
    return data.transformer.project(X_new), y_new


def gan_sample(
    generator: Generator,
    data: PreparedData,
    plan: dict[int, int],
    seed: int,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    device = next(generator.parameters()).device
    labels = np.concatenate([np.full(n, c) for c, n in plan.items()])
    chunks = [
        generator.sample(torch.as_tensor(labels[i : i + batch_size], device=device)).cpu().numpy()
        for i in range(0, len(labels), batch_size)
    ]
    return data.transformer.project(np.vstack(chunks)), labels


def ctgan_sample(
    data: PreparedData,
    plan: dict[int, int],
    seed: int,
    epochs: int,
    batch_size: int,
    max_rows_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from ctgan import CTGAN
    except ImportError as exc:  # optional dependency
        raise ImportError(
            "CTGAN baseline needs `pip install ctgan` (or `pip install .[ctgan]`)"
        ) from exc

    torch.manual_seed(seed)
    # CTGAN's epoch time grows with the number of rows. Large classes are subsampled
    # (every row of the smaller classes, including all rare ones, is kept), which keeps
    # one fit under an hour without changing what it learns about the rare classes.
    idx = np.arange(len(data.y_train))
    if max_rows_per_class:
        rng = np.random.default_rng(seed)
        idx = np.concatenate(
            [
                rows
                if len(rows) <= max_rows_per_class
                else rng.choice(rows, max_rows_per_class, replace=False)
                for rows in (np.flatnonzero(data.y_train == c) for c in range(data.n_classes))
            ]
        )
    frame = data.transformer.inverse_transform(data.X_train[idx])
    frame["label"] = np.asarray(data.classes)[data.y_train[idx]]
    discrete = [*data.transformer.categorical, *data.transformer.discrete, "label"]
    # ctgan >= 0.11 renamed `cuda` to `enable_gpu`.
    gpu_arg = "enable_gpu" if "enable_gpu" in inspect.signature(CTGAN).parameters else "cuda"
    model = CTGAN(
        epochs=epochs, batch_size=batch_size, verbose=True, **{gpu_arg: torch.cuda.is_available()}
    )
    model.set_random_state(seed)
    model.fit(frame, discrete_columns=discrete)

    parts, labels = [], []
    for c, n in plan.items():
        sample = model.sample(n, condition_column="label", condition_value=data.classes[c])
        parts.append(data.transformer.transform(sample.drop(columns="label")))
        labels.append(np.full(n, c))
    return np.vstack(parts), np.concatenate(labels)


def augment(
    method: str,
    data: PreparedData,
    plan: dict[int, int],
    seed: int,
    generator: Generator | None = None,
    ctgan_epochs: int = 300,
    ctgan_batch_size: int = 500,
    ctgan_max_rows_per_class: int | None = None,
) -> Augmented:
    if method == "none" or (method in GENERATIVE and not plan):
        return Augmented(data.X_train, data.y_train)
    if method == "class_weight":
        return Augmented(
            data.X_train, data.y_train, compute_sample_weight("balanced", data.y_train)
        )
    if method == "ros":
        return combine(data, *random_oversample(data, plan, seed))
    if method == "smote":
        return combine(data, *smote(data, plan, seed))
    if method == "ctgan":
        return combine(
            data,
            *ctgan_sample(
                data, plan, seed, ctgan_epochs, ctgan_batch_size, ctgan_max_rows_per_class
            ),
        )
    if method == "gan":
        if generator is None:
            raise ValueError("method 'gan' needs a trained generator")
        return combine(data, *gan_sample(generator, data, plan, seed))
    raise ValueError(f"Unknown augmentation method {method!r}")
