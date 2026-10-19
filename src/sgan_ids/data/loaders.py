"""Raw dataset loaders.

Each loader returns ``(frame, categorical_columns, official_test_mask)`` where
``frame`` has one column per feature plus a string ``label`` column. Identifier
columns (row ids, IPs, flow ids, timestamps) are removed here: they identify a
capture, not a behaviour, and a model that uses them would not transfer to new
traffic.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = "label"

# Columns that identify a specific capture rather than describe a flow.
_IDENTIFIER_COLUMNS = {
    "id",
    "flow id",
    "flow_id",
    "source ip",
    "src ip",
    "srcip",
    "destination ip",
    "dst ip",
    "dstip",
    "source port",
    "src port",
    "sport",
    "timestamp",
    "stime",
    "ltime",
}


def _clean_label(value: object) -> str:
    text = str(value)
    # CIC-IDS2017 stores an en dash as byte 0x96 (cp1252); normalise every dash variant.
    text = text.replace("\x96", "-").replace("–", "-").replace("�", "-")
    return re.sub(r"\s+", " ", text).strip()


def _drop_identifiers(frame: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in frame.columns if c.lower() in _IDENTIFIER_COLUMNS]
    return frame.drop(columns=drop)


def load_unsw_nb15(root: Path) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """UNSW-NB15 official partitions (175,341 train / 82,332 test flows).

    Uses the multi-class ``attack_cat`` column as the label; the binary ``label``
    column is dropped because it is a deterministic function of ``attack_cat``.
    """
    folder = root / "unsw_nb15"
    train_path = folder / "UNSW_NB15_training-set.csv"
    test_path = folder / "UNSW_NB15_testing-set.csv"
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. See scripts/download_data.md")

    parts = []
    for path, is_test in ((train_path, False), (test_path, True)):
        part = pd.read_csv(path, low_memory=False)
        part.columns = [c.strip().lower() for c in part.columns]
        part["__is_test"] = is_test
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)

    frame = frame.drop(columns=["label"])
    frame = frame.rename(columns={"attack_cat": LABEL})
    frame[LABEL] = (
        frame[LABEL].fillna("Normal").map(_clean_label).replace({"Backdoors": "Backdoor"})
    )
    frame = _drop_identifiers(frame)

    is_test = frame.pop("__is_test").to_numpy()
    categorical = [c for c in ("proto", "service", "state") if c in frame.columns]
    return frame, categorical, is_test


def load_cic_ids2017(root: Path) -> tuple[pd.DataFrame, list[str], None]:
    """CIC-IDS2017 ``MachineLearningCVE`` CSVs (8 files, one per capture session).

    There is no official train/test split, so ``official_test_mask`` is None and
    a stratified split is made later.
    """
    folder = root / "cic_ids2017"
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}. See scripts/download_data.md")

    parts = []
    for path in files:
        # Labels contain a cp1252 en dash, so utf-8 decoding fails on some files.
        part = pd.read_csv(path, encoding="latin-1", low_memory=False)
        part.columns = [c.strip() for c in part.columns]
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)

    frame = frame.rename(columns={"Label": LABEL})
    # "Fwd Header Length" appears twice in the original export.
    frame = frame.drop(columns=[c for c in frame.columns if c.endswith(".1")])
    frame = _drop_identifiers(frame)
    frame[LABEL] = frame[LABEL].map(_clean_label)

    features = [c for c in frame.columns if c != LABEL]
    frame[features] = frame[features].apply(pd.to_numeric, errors="coerce")
    frame[features] = frame[features].replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna().reset_index(drop=True)

    # Destination port is a service identifier, not a magnitude.
    frame["Destination Port"] = frame["Destination Port"].astype(int).astype(str)
    return frame, ["Destination Port"], None


def load_toy(
    root: Path | None = None, n_rows: int = 6000, seed: int = 0
) -> tuple[pd.DataFrame, list[str], None]:
    """Small synthetic dataset with the same shape as the real ones (used by tests and smoke runs).

    Four classes with a 60/30/8/2 % split, heavy-tailed count features, a
    categorical protocol column and an integer flag.
    """
    rng = np.random.default_rng(seed)
    shares = {"Normal": 0.60, "DoS": 0.30, "Recon": 0.08, "Worm": 0.02}
    rows = []
    for k, (name, share) in enumerate(shares.items()):
        n = int(n_rows * share)
        rows.append(
            pd.DataFrame(
                {
                    "duration": rng.exponential(1.0 + k, n),
                    "bytes": np.round(rng.lognormal(4 + k, 1.0, n)),
                    "packets": rng.poisson(3 + 4 * k, n),
                    "rate": rng.normal(k, 1.0, n),
                    "proto": rng.choice(["tcp", "udp", "icmp"], n, p=np.roll([0.7, 0.2, 0.1], k)),
                    "flag": rng.integers(0, 2, n) * (k % 2),
                    "constant": np.zeros(n),
                    LABEL: name,
                }
            )
        )
    return pd.concat(rows, ignore_index=True), ["proto"], None


LOADERS = {
    "unsw_nb15": load_unsw_nb15,
    "cic_ids2017": load_cic_ids2017,
    "toy": load_toy,
}
