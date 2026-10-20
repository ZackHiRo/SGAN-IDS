"""Experiment configuration.

Every run is fully described by one YAML file (see ``configs/``). The YAML is
parsed into nested dataclasses so typos fail loudly instead of being ignored,
and the resolved config is saved next to the results for reproducibility.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset: str = "unsw_nb15"  # "unsw_nb15" | "cic_ids2017" | "toy"
    root: str = "datasets"
    val_size: float = 0.15
    test_size: float = 0.15  # ignored for datasets with an official test split
    deduplicate: bool = True
    # Cap on rows per class *before* splitting (keeps the very large benign / DoS
    # classes of CIC-IDS2017 tractable). None = no cap.
    max_rows_per_class: int | None = None
    # Merge or drop labels after normalisation, e.g. {"Web Attack - XSS": "Web Attack"}.
    label_map: dict[str, str] = field(default_factory=dict)
    drop_classes: list[str] = field(default_factory=list)
    drop_features: list[str] = field(default_factory=list)
    # One-hot categories rarer than this fraction of training rows are grouped
    # into a single "infrequent" category.
    categorical_min_frequency: float = 0.001
    # Numeric columns with at most this many distinct training values (TTLs, window
    # sizes, flags) are encoded and generated as categories. 0 disables.
    max_discrete_values: int = 32
    # Exact values that make up >= mode_min_share of some class (and >= mode_min_rows rows)
    # get their own categorical output, so they can be generated exactly (e.g. jitter = 0).
    mode_min_share: float = 0.05
    mode_min_rows: int = 10
    max_modes_per_column: int = 16  # 0 disables


@dataclass
class GANConfig:
    latent_dim: int = 128
    class_embed_dim: int = 32
    generator_hidden: list[int] = field(default_factory=lambda: [256, 256, 256])
    critic_hidden: list[int] = field(default_factory=lambda: [256, 256])
    critic_dropout: float = 0.0
    # "projection" (Miyato & Koyama, 2018) or "concat" (label one-hot appended to input).
    conditioning: str = "projection"
    gumbel_tau: float = 0.2
    batch_size: int = 512
    n_steps: int = 20000  # generator updates
    n_critic: int = 5
    gp_lambda: float = 10.0
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    betas: tuple[float, float] = (0.5, 0.9)
    # Class sampling during training: p(c) ~ n_c ** alpha. 0 = uniform, 1 = empirical.
    class_sampling_alpha: float = 0.5
    ema_decay: float = 0.999  # 0 disables EMA
    eval_every: int = 1000
    eval_samples_per_class: int = 500
    device: str = "auto"


@dataclass
class AugmentConfig:
    # The rare classes under study: only these are augmented, and "rare-class" metrics
    # are averaged over exactly these. Empty = every class below target_per_class.
    classes: list[str] = field(default_factory=list)
    # Each augmented class with fewer training rows than this is topped up to this count.
    target_per_class: int = 10000
    methods: list[str] = field(
        default_factory=lambda: ["none", "class_weight", "ros", "smote", "ctgan", "gan"]
    )
    ctgan_epochs: int = 300
    ctgan_batch_size: int = 500
    # CTGAN is fitted on at most this many rows per class (rare classes are kept whole).
    ctgan_max_rows_per_class: int | None = 20000


@dataclass
class EvalConfig:
    classifiers: list[str] = field(default_factory=lambda: ["rf", "hgb", "mlp"])
    fidelity_max_samples: int = 2000  # per class, for kNN / C2ST metrics
    knn_k: int = 5
    bootstrap: int = 1000


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    output_dir: str = "artifacts"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    data: DataConfig = field(default_factory=DataConfig)
    gan: GANConfig = field(default_factory=GANConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls: type, values: dict[str, Any]) -> Any:
    """Recursively build a dataclass, rejecting unknown keys."""
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for key, value in values.items():
        default = (
            known[key].default_factory()
            if known[key].default_factory is not dataclasses.MISSING
            else known[key].default
        )
        if dataclasses.is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _build(type(default), value)
        elif isinstance(default, tuple) and isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _apply_override(raw: dict[str, Any], override: str) -> None:
    """Apply a ``a.b.c=value`` override; the value is parsed as YAML."""
    key, sep, value = override.partition("=")
    if not sep:
        raise ValueError(f"Override must look like key=value, got {override!r}")
    node = raw
    *parents, leaf = key.strip().split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = yaml.safe_load(value)


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None
) -> ExperimentConfig:
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    for override in overrides or []:
        _apply_override(raw, override)
    return _build(ExperimentConfig, raw)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    data = cfg.to_dict()
    data["gan"]["betas"] = list(data["gan"]["betas"])
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
