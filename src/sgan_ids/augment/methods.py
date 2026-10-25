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
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_sample_weight

from sgan_ids.data.prepare import PreparedData
from sgan_ids.models.networks import Generator
from sgan_ids.utils import logger

GENERATIVE = ("ros", "smote", "ctgan", "gan")


@dataclass
class Augmented:
    X: np.ndarray
    y: np.ndarray
    sample_weight: np.ndarray | None = None
    X_synth: np.ndarray | None = None
    y_synth: np.ndarray | None = None
    # Per-class notes from the generator (e.g. CTGAN's conditional hit rate).
    info: list[dict] = field(default_factory=list)


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
    # Decode from generation space to raw values, then encode into evaluation space.
    raw = data.gen_transformer.inverse_transform(np.vstack(chunks))
    return data.transformer.transform(raw), labels


def _new_ctgan(CTGAN, epochs: int, batch_size: int, seed: int):
    # ctgan >= 0.11 renamed `cuda` to `enable_gpu`.
    gpu_arg = "enable_gpu" if "enable_gpu" in inspect.signature(CTGAN).parameters else "cuda"
    model = CTGAN(
        epochs=epochs, batch_size=batch_size, verbose=False, **{gpu_arg: torch.cuda.is_available()}
    )
    model.set_random_state(seed)
    return model


def ctgan_sample(
    data: PreparedData,
    plan: dict[int, int],
    seed: int,
    epochs: int = 300,
    batch_size: int = 500,
    max_rows_per_class: int | None = None,
    mode: str = "per_class",
    min_steps: int = 3000,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """CTGAN baseline.

    ``per_class`` (default) fits one CTGAN on the training rows of each augmented class
    and samples from it unconditionally. ``conditional`` fits a single CTGAN on all
    classes and conditions on the label; runs 3-5 showed it generates the requested rare
    class for only 0.05-3 % of draws, and even those rows look like other classes.
    """
    try:
        from ctgan import CTGAN
    except ImportError as exc:  # optional dependency
        raise ImportError(
            "CTGAN baseline needs `pip install ctgan` (or `pip install .[ctgan]`)"
        ) from exc

    torch.manual_seed(seed)
    if mode == "conditional":
        return _ctgan_joint(CTGAN, data, plan, seed, epochs, batch_size, max_rows_per_class)
    if mode != "per_class":
        raise ValueError(f"Unknown ctgan mode {mode!r}")

    parts, labels, info = [], [], []
    for c, n in plan.items():
        frame = data.transformer.inverse_transform(data.X_train[data.y_train == c])
        # Columns that take a single value within this class cannot be modelled as continuous.
        constant = [col for col in frame.columns if frame[col].nunique() <= 1]
        discrete = sorted(
            {*data.transformer.categorical, *data.gen_transformer.discrete, *constant}
        )
        # CTGAN needs a batch size that is a multiple of its pac size (10) and at most the
        # number of rows. Small classes get more epochs so that every class is trained for
        # at least `min_steps` generator updates (108 Worms rows = one batch per epoch).
        bs = max(10, min(batch_size, len(frame)) // 10 * 10)
        steps_per_epoch = max(len(frame) // bs, 1)
        class_epochs = max(epochs, -(-min_steps // steps_per_epoch))
        model = _new_ctgan(CTGAN, class_epochs, bs, seed)
        model.fit(frame, discrete_columns=[col for col in discrete if col in frame.columns])
        parts.append(data.transformer.transform(model.sample(n)))
        labels.append(np.full(n, c))
        info.append(
            {
                "class": data.classes[c],
                "requested": n,
                "generated": n,
                "fit_rows": len(frame),
                "epochs": class_epochs,
                "batch_size": bs,
            }
        )
        logger.info(
            "  CTGAN for %s: %d rows, %d epochs, batch %d",
            data.classes[c],
            len(frame),
            class_epochs,
            bs,
        )
    if not parts:
        return np.empty((0, data.X_train.shape[1]), np.float32), np.empty(0, np.int64), []
    return np.vstack(parts), np.concatenate(labels), info


def _ctgan_joint(CTGAN, data, plan, seed, epochs, batch_size, max_rows_per_class):
    """One CTGAN for all classes, conditioned on the label (kept as a documented baseline)."""
    # CTGAN's epoch time grows with the number of rows. Large classes are subsampled
    # (every row of the smaller classes, including all rare ones, is kept).
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
    # Low-cardinality numerics are categories for CTGAN too, as they are for our GAN.
    discrete = [*data.transformer.categorical, *data.gen_transformer.discrete, "label"]
    model = _new_ctgan(CTGAN, epochs, batch_size, seed)
    model.fit(frame, discrete_columns=discrete)

    parts, labels, info = [], [], []
    for c, n in plan.items():
        rows, stats = _ctgan_conditional(model, data.classes[c], n)
        info.append({"class": data.classes[c], **stats})
        if len(rows):
            parts.append(data.transformer.transform(rows.drop(columns="label")))
            labels.append(np.full(len(rows), c))
    if not parts:
        return np.empty((0, data.X_train.shape[1]), np.float32), np.empty(0, np.int64), info
    return np.vstack(parts), np.concatenate(labels), info


def _ctgan_conditional(model, label: str, n: int, max_draw_factor: int = 200):
    """Draw ``n`` rows whose generated label really is ``label`` (rejection sampling).

    ``CTGAN.sample(condition_column=..., condition_value=...)`` only feeds the
    condition to the generator; the label it generates can still differ, and for
    rare classes it often does. Keeping every row would label other classes'
    traffic as the rare class. SDV's conditional sampling rejects mismatches in
    the same way. At most ``max_draw_factor * n`` rows are drawn; if fewer than
    ``n`` match, the class gets fewer synthetic rows and the shortfall is reported.
    """
    kept, drawn, matched = [], 0, 0
    batch = max(2 * n, 1000)
    while matched < n and drawn < max_draw_factor * n:
        sample = model.sample(batch, condition_column="label", condition_value=label)
        drawn += len(sample)
        hit = sample[sample["label"] == label]
        kept.append(hit)
        matched += len(hit)
    rows = pd.concat(kept).head(n) if kept else pd.DataFrame()
    stats = {
        "requested": n,
        "generated": len(rows),
        "drawn": drawn,
        "condition_hit_rate": matched / max(drawn, 1),
    }
    if len(rows) < n:
        logger.warning(
            "CTGAN produced only %d of %d '%s' rows (hit rate %.4f)",
            len(rows),
            n,
            label,
            stats["condition_hit_rate"],
        )
    return rows, stats


def augment(
    method: str,
    data: PreparedData,
    plan: dict[int, int],
    seed: int,
    generator: Generator | None = None,
    ctgan_options: dict | None = None,
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
        X, y, info = ctgan_sample(data, plan, seed, **(ctgan_options or {}))
        aug = combine(data, X, y)
        aug.info = info
        return aug
    if method == "gan":
        if generator is None:
            raise ValueError("method 'gan' needs a trained generator")
        return combine(data, *gan_sample(generator, data, plan, seed))
    raise ValueError(f"Unknown augmentation method {method!r}")
