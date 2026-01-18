from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

__all__ = ["DataForge", "DataSplit"]


@dataclass
class DataSplit:
    """Container for train/val/test splits with metadata."""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    n_features: int
    n_classes: int
    class_weights: np.ndarray  # For weighted sampling
    label_encoder: Optional[LabelEncoder] = None
    
    def get_class_distribution(self, split: str = "train") -> Dict[int, int]:
        """Get class distribution for a split."""
        y = getattr(self, f"y_{split}")
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))


class DataForge:
    """Dataset loader + preprocessing for network intrusion detection datasets.
    
    Supports:
    - NSL-KDD (legacy)
    - CIC-IDS2017 (legacy)
    - CIC-IDS2018 (modern)
    - UNSW-NB15 (modern)
    
    Features:
    - Proper train/val/test splits with stratification
    - Scaler fitted on training data only
    - Class weight computation for imbalanced learning
    """

    DATASET_LOADERS = {
        "nsl_kdd": "_load_nsl_kdd",
        "cic_ids2017": "_load_cic_ids2017",
        "cic_ids2018": "_load_cic_ids2018",
        "unsw_nb15": "_load_unsw_nb15",
    }

    def __init__(self, root, max_samples: Optional[int] = None):
        self.root = Path(root)
        self.frames: Dict[str, pd.DataFrame] = {}
        self.label_encoder: Optional[LabelEncoder] = None
        self.column_transformer: Optional[ColumnTransformer] = None
        self.max_samples = max_samples  # For memory-constrained environments

    # ----------------------- loading -----------------------
    def load(self, datasets: List[str] | None = None) -> "DataForge":
        """Load one or more datasets."""
        loaded = []
        for name in (datasets or self.DATASET_LOADERS):
            if name not in self.DATASET_LOADERS:
                print(f"Warning: Unknown dataset '{name}', skipping.")
                continue
            try:
                df = getattr(self, self.DATASET_LOADERS[name])()
                if df is not None and not df.empty:
                    self.frames[name] = df
                    loaded.append(name)
                else:
                    print(f"Warning: {name} loaded but is empty.")
            except FileNotFoundError as e:
                print(f"Warning: Could not load {name}: {e}")
            except Exception as e:
                print(f"Warning: Error loading {name}: {e}")
        if not loaded:
            raise RuntimeError("No datasets loaded successfully.")
        print(f"[DataForge] Loaded datasets: {loaded}")
        return self

    def _load_nsl_kdd(self) -> pd.DataFrame:
        """Load NSL-KDD dataset."""
        path = self.root / "NSL KDD/nsl-kdd/KDDTrain+.txt"
        if not path.exists():
            # Try alternative paths
            alt_paths = [
                self.root / "NSL-KDD/KDDTrain+.txt",
                self.root / "nsl_kdd/KDDTrain+.txt",
            ]
            for alt in alt_paths:
                if alt.exists():
                    path = alt
                    break
            else:
                raise FileNotFoundError(f"NSL-KDD not found at {path}")
        
        cols = [f"f{i}" for i in range(41)] + ["label", "difficulty"]
        df = pd.read_csv(path, names=cols)
        print(f"[NSL-KDD] Loaded shape: {df.shape}")
        return df

    def _load_cic_ids2017(self) -> pd.DataFrame:
        """Load CIC-IDS2017 dataset."""
        fold = self.root / "CIC-IDS_2017/MachineLearningCSV/MachineLearningCVE"
        if not fold.exists():
            fold = self.root / "CIC-IDS2017"
        files = list(fold.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {fold}")
        
        max_samples = self.max_samples
        dfs = []
        total_loaded = 0
        
        for f in files:
            try:
                if max_samples:
                    # Use chunked reading to limit memory usage
                    chunk_list = []
                    chunk_size = 50000
                    for chunk in pd.read_csv(f, chunksize=chunk_size, low_memory=False):
                        chunk_list.append(chunk)
                        total_loaded += len(chunk)
                        if total_loaded >= max_samples * 1.5:
                            break
                    if chunk_list:
                        dfs.append(pd.concat(chunk_list, ignore_index=True))
                    if total_loaded >= max_samples * 1.5:
                        print(f"[CIC-IDS2017] Early stop after {len(dfs)} files, ~{total_loaded} rows")
                        break
                else:
                    df = pd.read_csv(f, low_memory=False)
                    dfs.append(df)
            except Exception as e:
                print(f"Warning: Error reading {f.name}: {e}")
        
        if not dfs:
            raise RuntimeError("Failed to load any CIC-IDS2017 files")
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Sample down if we exceeded limit
        if max_samples and len(df) > max_samples:
            print(f"[CIC-IDS2017] Sampling {max_samples} from {len(df)} rows")
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
        # Standardize label column name
        if " Label" in df.columns:
            df = df.rename(columns={" Label": "label"})
        elif "Label" in df.columns:
            df = df.rename(columns={"Label": "label"})
        
        print(f"[CIC-IDS2017] Loaded shape: {df.shape}")
        return df

    def _load_cic_ids2018(self) -> pd.DataFrame:
        """Load CIC-IDS2018 dataset."""
        fold = self.root / "CIC-IDS2018"
        if not fold.exists():
            fold = self.root / "CSE-CIC-IDS2018"
        
        files = list(fold.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {fold}")
        
        max_samples = self.max_samples
        dfs = []
        total_loaded = 0
        
        for f in files:
            try:
                if max_samples:
                    # Use chunked reading to limit memory usage
                    chunk_list = []
                    chunk_size = 50000
                    for chunk in pd.read_csv(f, chunksize=chunk_size, low_memory=False):
                        chunk_list.append(chunk)
                        total_loaded += len(chunk)
                        # Stop if we have enough data (with some buffer for class balance)
                        if total_loaded >= max_samples * 1.5:
                            break
                    if chunk_list:
                        dfs.append(pd.concat(chunk_list, ignore_index=True))
                    # Check if we have enough total
                    if total_loaded >= max_samples * 1.5:
                        print(f"[CIC-IDS2018] Early stop after {len(dfs)} files, ~{total_loaded} rows")
                        break
                else:
                    df = pd.read_csv(f, low_memory=False)
                    dfs.append(df)
            except Exception as e:
                print(f"Warning: Error reading {f.name}: {e}")
        
        if not dfs:
            raise RuntimeError("Failed to load any CIC-IDS2018 files")
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Sample down if we exceeded limit
        if max_samples and len(df) > max_samples:
            print(f"[CIC-IDS2018] Sampling {max_samples} from {len(df)} rows")
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
        # Standardize label column
        if "Label" in df.columns:
            df = df.rename(columns={"Label": "label"})
        
        print(f"[CIC-IDS2018] Loaded shape: {df.shape}")
        return df

    def _load_unsw_nb15(self) -> pd.DataFrame:
        """Load UNSW-NB15 dataset."""
        fold = self.root / "UNSW-NB15"
        if not fold.exists():
            fold = self.root / "unsw_nb15"
        
        # Try to find training file
        train_files = list(fold.glob("*[Tt]rain*.csv")) + list(fold.glob("UNSW_NB15_training*.csv"))
        test_files = list(fold.glob("*[Tt]est*.csv")) + list(fold.glob("UNSW_NB15_testing*.csv"))
        
        files = train_files + test_files
        if not files:
            # Try generic CSV files
            files = list(fold.glob("*.csv"))
        
        if not files:
            raise FileNotFoundError(f"No CSV files found in {fold}")
        
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f, low_memory=False)
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Error reading {f.name}: {e}")
        
        if not dfs:
            raise RuntimeError("Failed to load any UNSW-NB15 files")
        
        df = pd.concat(dfs, ignore_index=True)
        
        # UNSW-NB15 has 'attack_cat' for attack category and 'label' for binary
        # We'll use attack_cat if available for multi-class
        if "attack_cat" in df.columns:
            df = df.rename(columns={"attack_cat": "label"})
        elif "Label" in df.columns:
            df = df.rename(columns={"Label": "label"})
        
        print(f"[UNSW-NB15] Loaded shape: {df.shape}")
        return df

    # -------------------- preprocessing --------------------
    def preprocess(
        self,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[DataSplit, ColumnTransformer]:
        """Preprocess data with proper train/val/test splits.
        
        Key improvements:
        - Stratified splitting to preserve class distribution
        - Scaler fitted on training data only (no data leakage)
        - Returns DataSplit object with all splits and metadata
        
        Note: max_samples is now handled during loading via DataForge constructor.
        
        Args:
            val_size: Fraction of data for validation
            test_size: Fraction of data for test
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (DataSplit, ColumnTransformer)
        """
        if not self.frames:
            raise RuntimeError("No data loaded. Did you call .load()?")

        # --- Feature alignment ---
        all_cols = set()
        for df in self.frames.values():
            all_cols.update([c for c in df.columns if c not in {"label", "difficulty", "id"}])
        all_cols = sorted(all_cols)
        all_cols.append("label")

        aligned_frames = []
        for name, df in self.frames.items():
            df = df.copy()
            # Drop non-feature columns
            for col in ["difficulty", "id"]:
                if col in df.columns:
                    df = df.drop(columns=[col])
            # Add missing columns as zeros
            missing = set(all_cols) - set(df.columns)
            for col in missing:
                if col != "label":
                    df[col] = 0
            # Ensure column order
            df = df[[c for c in all_cols if c in df.columns]]
            aligned_frames.append(df)

        df = pd.concat(aligned_frames, ignore_index=True).drop_duplicates()

        # --- Clean data ---
        df = df.replace(["Infinity", "NaN", "nan", "inf", np.inf, -np.inf], np.nan)
        initial_size = len(df)
        df = df.dropna()
        print(f"[DataForge] Dropped {initial_size - len(df)} rows with NaN/inf values")
        print(f"[DataForge] Data shape after cleaning: {df.shape}")

        # --- Separate features and labels ---
        import gc
        y = df["label"].values
        df_features = df.drop(columns=["label"])
        del df  # Free memory early
        gc.collect()

        # --- Encode labels ---
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)
        print(f"[DataForge] Number of classes: {n_classes}")
        print(f"[DataForge] Classes: {list(self.label_encoder.classes_)}")

        # --- Identify column types ---
        cat_cols = df_features.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df_features.select_dtypes(exclude=["object", "category"]).columns.tolist()

        # --- Train/Val/Test split (BEFORE fitting scaler) ---
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            df_features, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
        
        # Second split: separate validation from training
        val_ratio = val_size / (1 - test_size)  # Adjust ratio for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            stratify=y_temp,
            random_state=random_state,
        )

        print(f"[DataForge] Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # --- Build and fit transformer on TRAINING data only ---
        import gc
        
        transformers = []
        if cat_cols:
            # Use sparse=True for memory efficiency during fitting
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols))
        if num_cols:
            transformers.append(("num", StandardScaler(), num_cols))
        
        if not transformers:
            raise RuntimeError("No valid columns to encode.")

        ct = ColumnTransformer(transformers, sparse_threshold=0.3)
        
        # Fit on training data only!
        print("[DataForge] Fitting transformers on training data...")
        X_train_transformed = ct.fit_transform(X_train)
        
        # Free memory before transforming val/test
        del X_train
        gc.collect()
        
        print("[DataForge] Transforming validation data...")
        X_val_transformed = ct.transform(X_val)
        del X_val
        gc.collect()
        
        print("[DataForge] Transforming test data...")
        X_test_transformed = ct.transform(X_test)
        del X_test
        gc.collect()
        
        self.column_transformer = ct

        # Convert sparse matrices to dense if needed (do incrementally to save memory)
        if hasattr(X_train_transformed, "toarray"):
            print("[DataForge] Converting sparse to dense...")
            X_train_transformed = X_train_transformed.toarray()
            gc.collect()
            X_val_transformed = X_val_transformed.toarray()
            gc.collect()
            X_test_transformed = X_test_transformed.toarray()
            gc.collect()

        # --- Compute class weights for weighted sampling ---
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum()

        n_features = X_train_transformed.shape[1]
        print(f"[DataForge] Number of features after encoding: {n_features}")

        data_split = DataSplit(
            X_train=X_train_transformed.astype(np.float32),
            X_val=X_val_transformed.astype(np.float32),
            X_test=X_test_transformed.astype(np.float32),
            y_train=y_train.astype(np.int64),
            y_val=y_val.astype(np.int64),
            y_test=y_test.astype(np.int64),
            n_features=n_features,
            n_classes=n_classes,
            class_weights=class_weights.astype(np.float32),
            label_encoder=self.label_encoder,
        )

        return data_split, ct

    def minority_classes(self, thr: float = 0.01) -> List[str]:
        """Get list of minority class names (below threshold frequency)."""
        df = pd.concat(self.frames.values())
        return (
            df["label"].value_counts(normalize=True).loc[lambda s: s < thr].index.tolist()
        )

    def get_label_mapping(self) -> Optional[Dict[str, int]]:
        """Return dict mapping original label string -> integer id."""
        if self.label_encoder is None:
            return None
        return {cls: idx for idx, cls in enumerate(self.label_encoder.classes_)} 