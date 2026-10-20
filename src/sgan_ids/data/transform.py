"""Reversible tabular encoding shared by the GAN, the baselines and the classifiers.

Numeric features
    Flow statistics are heavy-tailed (bytes, durations and rates span 6+ orders
    of magnitude), so columns whose magnitude exceeds 100 are passed through a
    signed ``log1p``. Every numeric column is then min-max scaled to [-1, 1]
    using training statistics only. The generator's numeric outputs use ``tanh``,
    so its output range matches the data range exactly.

Categorical features
    One-hot encoded; categories rarer than ``min_frequency`` are grouped. The
    generator emits each categorical block through a Gumbel-softmax.

Discrete numeric features
    Numeric columns with at most ``max_discrete_values`` distinct training values
    (TTLs, TCP window sizes, flags, small counters) are one-hot encoded over the
    values actually observed, without grouping. Generated as a continuous number
    and rounded, such a column lands on values that never occur: in the first
    UNSW-NB15 run the GAN gave Shellcode ``sttl=255`` in half its rows, while every
    real Shellcode flow has 254. That single artefact made synthetic rows trivially
    separable from real ones. As categories, only observed values can be produced.

Frequent exact values ("modes") of continuous features
    Many flow statistics put a large share of rows on one exact value: half of
    UNSW-NB15's Shellcode flows have jitter, RTT and load of exactly 0, and 45 %
    of Worms flows have ``dmean`` exactly 45. A continuous output never lands
    exactly on such a value, so even a well-trained generator was perfectly
    separable from real data. Following CTAB-GAN's "mixed" columns (Zhao et al.,
    2021), each continuous column also gets a categorical selector over its modes
    (values that make up at least ``mode_min_share`` of some class, and at least
    ``mode_min_rows`` rows) plus "other". When the selector picks a mode, the
    decoded value is exactly that value; otherwise the continuous output is used.

``project`` maps any vector (GAN output, SMOTE interpolation) back onto the set
of valid records: categorical blocks become exact one-hots, integer columns are
rounded in raw units, and values are clipped to the training range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

LOG_THRESHOLD = 100.0


@dataclass(frozen=True)
class Span:
    """A contiguous block of encoded columns."""

    start: int
    end: int
    kind: str  # "numeric" | "categorical"
    name: str


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _signed_expm1(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.expm1(np.abs(x))


class TabularTransformer:
    def __init__(
        self,
        categorical: list[str],
        min_frequency: float = 0.001,
        max_discrete_values: int = 32,
        mode_min_share: float = 0.05,
        mode_min_rows: int = 10,
        max_modes_per_column: int = 16,
    ):
        self.categorical = list(categorical)
        self.min_frequency = min_frequency
        self.max_discrete_values = max_discrete_values
        self.mode_min_share = mode_min_share
        self.mode_min_rows = mode_min_rows
        self.max_modes_per_column = max_modes_per_column

    # ------------------------------------------------------------------ fit
    def fit(self, frame: pd.DataFrame, y: np.ndarray | None = None) -> TabularTransformer:
        """Fit on training rows. ``y`` (class labels) lets modes be found per class."""
        numeric = [c for c in frame.columns if c not in self.categorical]
        values = frame[numeric].to_numpy(dtype=np.float64)

        # Constant columns carry no information and make min-max scaling undefined.
        n_unique = frame[numeric].nunique().to_numpy()
        keep = n_unique > 1
        self.dropped_constant = [c for c, k in zip(numeric, keep) if not k]
        self.discrete = [c for c, n in zip(numeric, n_unique) if 1 < n <= self.max_discrete_values]
        continuous = (n_unique > 1) & ~np.isin(numeric, self.discrete)
        self.numeric = [c for c, k in zip(numeric, continuous) if k]
        values = values[:, continuous]

        self.integer = np.all(np.equal(np.mod(values, 1), 0), axis=0)
        self.log = np.nanmax(np.abs(values), axis=0) > LOG_THRESHOLD
        scaled = np.where(self.log, _signed_log1p(values), values)
        self.raw_min = values.min(axis=0)
        self.raw_max = values.max(axis=0)
        self.min = scaled.min(axis=0)
        self.max = scaled.max(axis=0)

        self.encoder = None
        if self.categorical:
            self.encoder = OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=self.min_frequency,
                sparse_output=False,
                dtype=np.float32,
            ).fit(frame[self.categorical].astype(str))

        self.discrete_encoder = None
        if self.discrete:
            self.discrete_encoder = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, dtype=np.float32
            ).fit(self._discrete_strings(frame))

        spans = []
        for i, name in enumerate(self.numeric):
            spans.append(Span(i, i + 1, "numeric", name))
        offset = len(self.numeric)
        if self.encoder is not None:
            for name, cats in zip(self.categorical, self.encoder.categories_):
                width = (
                    len(cats) - len(self._infrequent(name)) + (1 if self._infrequent(name) else 0)
                )
                spans.append(Span(offset, offset + width, "categorical", name))
                offset += width
        self.n_categorical = offset - len(self.numeric)
        if self.discrete_encoder is not None:
            for name, cats in zip(self.discrete, self.discrete_encoder.categories_):
                spans.append(Span(offset, offset + len(cats), "categorical", name))
                offset += len(cats)
        self.n_discrete = offset - len(self.numeric) - self.n_categorical
        self.modes = self._find_modes(frame, y)
        for name, modes in zip(self.numeric, self.modes):
            if len(modes):
                spans.append(Span(offset, offset + len(modes) + 1, "categorical", f"{name}=mode"))
                offset += len(modes) + 1
        self.spans = spans
        self.n_features = offset
        return self

    def _find_modes(self, frame: pd.DataFrame, y: np.ndarray | None) -> list[np.ndarray]:
        groups = (
            [np.arange(len(frame))] if y is None else [np.flatnonzero(y == c) for c in np.unique(y)]
        )
        modes = []
        for name in self.numeric:
            col = frame[name].to_numpy(dtype=np.float64)
            found: set[float] = set()
            if self.max_modes_per_column > 0:
                for rows in groups:
                    vc = pd.Series(col[rows]).value_counts()
                    keep = (vc / len(rows) >= self.mode_min_share) & (vc >= self.mode_min_rows)
                    found |= set(vc.index[keep])
            # Keep the most frequent ones overall if there are too many.
            overall = pd.Series(col).value_counts()
            ranked = [v for v in overall.index if v in found][: self.max_modes_per_column]
            modes.append(np.array(sorted(ranked), dtype=np.float64))
        return modes

    def _discrete_strings(self, frame: pd.DataFrame) -> pd.DataFrame:
        # Through float64 first, so 254 (int) and 254.0 (float) map to the same category.
        return frame[self.discrete].astype(np.float64).astype(str)

    def _infrequent(self, name: str) -> list:
        idx = self.categorical.index(name)
        infrequent = self.encoder.infrequent_categories_[idx]
        return [] if infrequent is None else list(infrequent)

    # ------------------------------------------------------------ transform
    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.numeric].to_numpy(dtype=np.float64)
        values = np.where(self.log, _signed_log1p(values), values)
        numeric = 2.0 * (values - self.min) / (self.max - self.min) - 1.0
        numeric = np.clip(numeric, -1.0, 1.0)
        blocks = [numeric.astype(np.float32)]
        if self.encoder is not None:
            blocks.append(self.encoder.transform(frame[self.categorical].astype(str)))
        if self.discrete_encoder is not None:
            blocks.append(self.discrete_encoder.transform(self._discrete_strings(frame)))
        raw = frame[self.numeric].to_numpy(dtype=np.float64)
        for j, modes in enumerate(self.modes):
            if len(modes):
                idx = pd.Index(modes).get_indexer(raw[:, j])
                idx[idx < 0] = len(modes)  # "other"
                blocks.append(np.eye(len(modes) + 1, dtype=np.float32)[idx])
        return np.hstack(blocks).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> pd.DataFrame:
        n_num = len(self.numeric)
        numeric = np.clip(x[:, :n_num].astype(np.float64), -1.0, 1.0)
        values = (numeric + 1.0) / 2.0 * (self.max - self.min) + self.min
        values = np.where(self.log, _signed_expm1(values), values)
        values = np.clip(values, self.raw_min, self.raw_max)
        values = np.where(self.integer, np.round(values), values)
        frame = pd.DataFrame(values, columns=self.numeric)
        onehot = self.one_hot_categoricals(x)
        cat_end = n_num + self.n_categorical
        if self.encoder is not None:
            cats = self.encoder.inverse_transform(onehot[:, n_num:cat_end])
            for i, name in enumerate(self.categorical):
                frame[name] = cats[:, i]
        disc_end = cat_end + self.n_discrete
        if self.discrete_encoder is not None:
            vals = self.discrete_encoder.inverse_transform(onehot[:, cat_end:disc_end])
            for i, name in enumerate(self.discrete):
                frame[name] = vals[:, i].astype(np.float64)
        offset = disc_end
        for name, modes in zip(self.numeric, self.modes):
            if len(modes):
                choice = onehot[:, offset : offset + len(modes) + 1].argmax(axis=1)
                on_mode = choice < len(modes)
                frame.loc[on_mode, name] = modes[choice[on_mode]]
                offset += len(modes) + 1
        return frame

    # -------------------------------------------------------------- helpers
    def one_hot_categoricals(self, x: np.ndarray) -> np.ndarray:
        """Replace every categorical block by the one-hot of its argmax."""
        out = np.array(x, dtype=np.float32, copy=True)
        for span in self.spans:
            if span.kind == "categorical":
                block = out[:, span.start : span.end]
                hot = np.zeros_like(block)
                hot[np.arange(len(block)), block.argmax(axis=1)] = 1.0
                out[:, span.start : span.end] = hot
        return out

    def project(self, x: np.ndarray) -> np.ndarray:
        """Snap encoded vectors onto the domain of valid records."""
        return self.transform(self.inverse_transform(x))

    def feature_names(self) -> list[str]:
        names = list(self.numeric)
        if self.encoder is not None:
            names += list(self.encoder.get_feature_names_out(self.categorical))
        if self.discrete_encoder is not None:
            names += list(self.discrete_encoder.get_feature_names_out(self.discrete))
        for name, modes in zip(self.numeric, self.modes):
            if len(modes):
                names += [f"{name}={v:g}" for v in modes] + [f"{name}=other"]
        return names
