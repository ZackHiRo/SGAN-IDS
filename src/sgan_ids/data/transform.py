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
    def __init__(self, categorical: list[str], min_frequency: float = 0.001):
        self.categorical = list(categorical)
        self.min_frequency = min_frequency

    # ------------------------------------------------------------------ fit
    def fit(self, frame: pd.DataFrame) -> TabularTransformer:
        numeric = [c for c in frame.columns if c not in self.categorical]
        values = frame[numeric].to_numpy(dtype=np.float64)

        # Constant columns carry no information and make min-max scaling undefined.
        keep = np.nanmax(values, axis=0) > np.nanmin(values, axis=0)
        self.dropped_constant = [c for c, k in zip(numeric, keep) if not k]
        self.numeric = [c for c, k in zip(numeric, keep) if k]
        values = values[:, keep]

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
        self.spans = spans
        self.n_features = offset
        return self

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
        return np.hstack(blocks).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> pd.DataFrame:
        n_num = len(self.numeric)
        numeric = np.clip(x[:, :n_num].astype(np.float64), -1.0, 1.0)
        values = (numeric + 1.0) / 2.0 * (self.max - self.min) + self.min
        values = np.where(self.log, _signed_expm1(values), values)
        values = np.clip(values, self.raw_min, self.raw_max)
        values = np.where(self.integer, np.round(values), values)
        frame = pd.DataFrame(values, columns=self.numeric)
        if self.encoder is not None:
            onehot = self.one_hot_categoricals(x)[:, n_num:]
            cats = self.encoder.inverse_transform(onehot)
            for i, name in enumerate(self.categorical):
                frame[name] = cats[:, i]
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
        return names
