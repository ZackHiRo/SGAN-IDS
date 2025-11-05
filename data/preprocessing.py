from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

__all__ = ["DataForge"]


class DataForge:
    """Dataset loader + basic preprocessing for NSL-KDD & CIC-IDS2017."""

    DATASET_LOADERS = {
        "nsl_kdd": "_load_nsl_kdd",
        "cic_ids2017": "_load_cic_ids2017",
    }

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.frames: Dict[str, pd.DataFrame] = {}

    # ----------------------- loading -----------------------
    def load(self, datasets: List[str] | None = None):
        loaded = []
        for name in (datasets or self.DATASET_LOADERS):
            try:
                df = getattr(self, self.DATASET_LOADERS[name])()
                if not df.empty:
                    self.frames[name] = df
                    loaded.append(name)
                else:
                    print(f"Warning: {name} loaded but is empty.")
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")
        if not loaded:
            raise RuntimeError("No datasets loaded successfully.")
        return self

    def _load_nsl_kdd(self) -> pd.DataFrame:
        path = self.root / "NSL KDD/nsl-kdd/KDDTrain+.txt"
        cols = [f"f{i}" for i in range(41)] + ["label", "difficulty"]
        df = pd.read_csv(path, names=cols)
        print(f"[NSL-KDD] Loaded shape: {df.shape}")
        print(df.head())
        return df

    def _load_cic_ids2017(self) -> pd.DataFrame:
        fold = self.root / "CIC-IDS_2017/MachineLearningCSV/MachineLearningCVE"
        files = list(fold.glob("*.csv"))
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # -------------------- preprocessing --------------------
    def preprocess(self) -> Tuple[Tuple, Pipeline]:
        if not self.frames:
            raise RuntimeError("No data loaded. Did you call .load()?")

        # --- Feature alignment: align all datasets to the union of all columns (excluding label/difficulty) ---
        # Collect all columns (excluding label/difficulty)
        all_cols = set()
        for df in self.frames.values():
            all_cols.update([c for c in df.columns if c not in {"label", "difficulty"}])
        all_cols = sorted(all_cols)
        # Add label at the end
        all_cols.append("label")

        print("[DEBUG] Starting feature alignment loop...")
        aligned_frames = []
        for df in self.frames.values():
            # Drop 'difficulty' if present
            if 'difficulty' in df.columns:
                df = df.drop(columns=['difficulty'])
            # Add missing columns as zeros
            missing = set(all_cols) - set(df.columns)
            for col in missing:
                if col != "label":
                    df[col] = 0
            # Ensure column order
            df = df[[c for c in all_cols if c in df.columns]]
            aligned_frames.append(df)
        print("[DEBUG] Feature alignment loop done.")

        print("[DEBUG] Concatenating aligned frames...")
        df = pd.concat(aligned_frames).drop_duplicates()
        print("[DEBUG] Concatenation and drop_duplicates done.")

        print("[DEBUG] Cleaning NaNs and infs...")
        df.replace(["Infinity", "NaN", "nan", "inf"], pd.NA, inplace=True)
        df.dropna(inplace=True)
        print("[DEBUG] Cleaning done.")

        print(f"[DataForge] Data shape after cleaning: {df.shape}")
        print(f"[DataForge] Columns: {list(df.columns)}")

        # Separate features and labels to avoid encoding 'label' as a feature
        y = df["label"].values if "label" in df else None
        df_features = df.drop(columns=["label"]) if "label" in df.columns else df.copy()

        cat = df_features.select_dtypes(include=["object", "category"]).columns
        num = df_features.select_dtypes(exclude=["object", "category"]).columns

        print(f"[DEBUG] Categorical columns: {list(cat)}")
        print(f"[DEBUG] Numerical columns: {list(num)}")

        transformers = []
        if len(cat) > 0:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
        if len(num) > 0:
            transformers.append(("num", StandardScaler(), num))
        if not transformers:
            raise RuntimeError("No valid columns to encode (no categorical or numerical columns found).")

        print("[DEBUG] Fitting ColumnTransformer...")
        ct = ColumnTransformer(transformers)
        X = ct.fit_transform(df_features)
        print("[DEBUG] ColumnTransformer fit_transform done.")

        # Encode string labels to integer IDs for efficient training
        self.label_encoder = None
        if y is not None and y.dtype.kind in {"O", "U", "S"}:
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoder = le
        return (X, y), ct

    def minority_classes(self, thr: float = 0.01) -> List[str]:
        df = pd.concat(self.frames.values())
        return (
            df["label"].value_counts(normalize=True).loc[lambda s: s < thr].index.tolist()
        )

    # -------------------- helpers --------------------
    def get_label_mapping(self):
        """Return dict mapping original label string -> integer id."""
        if self.label_encoder is None:
            return None
        return {cls: idx for idx, cls in enumerate(self.label_encoder.classes_)} 