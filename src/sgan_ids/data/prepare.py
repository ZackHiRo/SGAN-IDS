"""Turn a raw dataset into fixed, encoded train / val / test splits.

The prepared splits are written to disk once and reused by every later step,
so the GAN, the baselines and the classifiers all see exactly the same rows.
The test split is used only for the final evaluation.

Two encodings of the same rows are kept:

- **Evaluation space** (``X_*``, ``transformer``): one column per numeric feature
  plus one-hot categoricals. Classifiers, SMOTE, random oversampling and every
  metric work here, so baselines do not depend on how the GAN represents data.
- **Generation space** (``G_*``, ``gen_transformer``): the evaluation space plus
  one-hot blocks for low-cardinality numerics and exact-value ("mode") selectors
  for continuous columns. Only the GAN trains and samples here; its output is
  decoded to raw values and re-encoded into evaluation space.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sgan_ids.config import DataConfig
from sgan_ids.data.loaders import LABEL, LOADERS
from sgan_ids.data.transform import TabularTransformer


@dataclass
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    classes: list[str]
    transformer: TabularTransformer
    report: dict
    gen_transformer: TabularTransformer | None = None
    G_train: np.ndarray | None = None
    G_val: np.ndarray | None = None
    folder: Path | None = None
    _fingerprint: str | None = None

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def generation_splits(self) -> tuple[np.ndarray, np.ndarray]:
        """Training and validation rows in generation space (loaded from disk on first use)."""
        if self.G_train is None:
            if self.folder is None or not (self.folder / "gan_splits.npz").exists():
                raise FileNotFoundError("Generation-space splits missing; run the prepare step.")
            arrays = np.load(self.folder / "gan_splits.npz")
            self.G_train, self.G_val = arrays["G_train"], arrays["G_val"]
        return self.G_train, self.G_val

    def release_generation_splits(self) -> None:
        """Free the (much wider) generation-space arrays once the GANs are trained."""
        if self.folder is not None:
            self.G_train = self.G_val = None

    def fingerprint(self) -> str:
        """Hash of the training split in both encodings; anything fitted on it is stale when it changes."""
        if self._fingerprint is None:
            h = hashlib.sha1()
            h.update(json.dumps(self.classes).encode())
            h.update(self.y_train.tobytes())
            h.update(np.ascontiguousarray(self.X_train).tobytes())
            h.update(np.ascontiguousarray(self.generation_splits()[0]).tobytes())
            self._fingerprint = h.hexdigest()
        return self._fingerprint

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / "splits.npz",
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            X_test=self.X_test,
            y_test=self.y_test,
        )
        np.savez_compressed(folder / "gan_splits.npz", G_train=self.G_train, G_val=self.G_val)
        joblib.dump(self.transformer, folder / "transformer.joblib")
        joblib.dump(self.gen_transformer, folder / "gen_transformer.joblib")
        self.folder = folder
        with open(folder / "data_report.json", "w") as f:
            json.dump({"classes": self.classes, **self.report}, f, indent=2)

    @classmethod
    def load(cls, folder: Path) -> PreparedData:
        arrays = np.load(folder / "splits.npz")
        with open(folder / "data_report.json") as f:
            report = json.load(f)
        classes = report.pop("classes")
        if not (folder / "gen_transformer.joblib").exists():
            raise FileNotFoundError(
                f"{folder} was prepared by an older version; run the prepare step again."
            )
        return cls(
            **{k: arrays[k] for k in arrays.files},
            classes=classes,
            transformer=joblib.load(folder / "transformer.joblib"),
            report=report,
            gen_transformer=joblib.load(folder / "gen_transformer.joblib"),
            folder=folder,
        )


def _counts(labels: pd.Series | np.ndarray, classes: list[str] | None = None) -> dict[str, int]:
    series = pd.Series(labels)
    if classes is not None:
        series = series.map(dict(enumerate(classes)))
    return {str(k): int(v) for k, v in series.value_counts().items()}


def prepare(cfg: DataConfig, seed: int = 0) -> PreparedData:
    root = Path(cfg.root)
    frame, categorical, official_test = LOADERS[cfg.dataset](root)
    report: dict = {"dataset": cfg.dataset, "raw_rows": len(frame)}

    # ---- labels ---------------------------------------------------------
    frame[LABEL] = frame[LABEL].replace(cfg.label_map)
    keep = ~frame[LABEL].isin(cfg.drop_classes)
    frame = frame[keep].reset_index(drop=True)
    if official_test is not None:
        official_test = official_test[keep.to_numpy()]
    report["dropped_classes"] = cfg.drop_classes

    frame = frame.drop(columns=[c for c in cfg.drop_features if c in frame.columns])
    categorical = [c for c in categorical if c in frame.columns]

    # ---- splits ---------------------------------------------------------
    if official_test is not None:
        train_full, test = frame[~official_test], frame[official_test]
        if cfg.deduplicate:
            before = len(train_full)
            train_full = train_full.drop_duplicates()
            report["train_duplicates_removed"] = before - len(train_full)
        report["split"] = "official"
    else:
        if cfg.deduplicate:
            before = len(frame)
            frame = frame.drop_duplicates()
            report["duplicates_removed"] = before - len(frame)
        if cfg.max_rows_per_class:
            cap = cfg.max_rows_per_class
            frame = pd.concat(
                [
                    group.sample(min(len(group), cap), random_state=seed)
                    for _, group in frame.groupby(LABEL)
                ]
            )
            report["max_rows_per_class"] = cfg.max_rows_per_class
        train_full, test = train_test_split(
            frame, test_size=cfg.test_size, stratify=frame[LABEL], random_state=seed
        )
        report["split"] = f"stratified test_size={cfg.test_size}"

    train, val = train_test_split(
        train_full, test_size=cfg.val_size, stratify=train_full[LABEL], random_state=seed
    )

    # ---- encoding (fit on train only) ------------------------------------
    classes = sorted(train[LABEL].unique())
    index = {c: i for i, c in enumerate(classes)}
    unseen = set(test[LABEL]) - set(classes)
    if unseen:
        raise ValueError(f"Test labels absent from training data: {sorted(unseen)}")

    features = [c for c in frame.columns if c != LABEL]
    # Evaluation space: no discrete one-hots, no mode selectors.
    transformer = TabularTransformer(
        categorical, cfg.categorical_min_frequency, max_discrete_values=0, max_modes_per_column=0
    ).fit(train[features])
    gen_transformer = TabularTransformer(
        categorical,
        cfg.categorical_min_frequency,
        cfg.max_discrete_values,
        cfg.mode_min_share,
        cfg.mode_min_rows,
        cfg.max_modes_per_column,
    ).fit(train[features], train[LABEL].to_numpy())

    def encode(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return transformer.transform(part[features]), part[LABEL].map(index).to_numpy(np.int64)

    X_train, y_train = encode(train)
    X_val, y_val = encode(val)
    X_test, y_test = encode(test)
    G_train = gen_transformer.transform(train[features])
    G_val = gen_transformer.transform(val[features])

    report.update(
        {
            "n_features_raw": len(features),
            "n_features_encoded": transformer.n_features,
            "n_features_generation": gen_transformer.n_features,
            "categorical": categorical,
            "discrete_numeric": gen_transformer.discrete,
            "modes_per_column": {
                c: len(m) for c, m in zip(gen_transformer.numeric, gen_transformer.modes) if len(m)
            },
            "log_scaled": [c for c, flag in zip(transformer.numeric, transformer.log) if flag],
            "dropped_constant": transformer.dropped_constant,
            "rows": {"train": len(train), "val": len(val), "test": len(test)},
            "class_counts": {
                "train": _counts(y_train, classes),
                "val": _counts(y_val, classes),
                "test": _counts(y_test, classes),
            },
        }
    )
    return PreparedData(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        classes,
        transformer,
        report,
        gen_transformer=gen_transformer,
        G_train=G_train,
        G_val=G_val,
    )
