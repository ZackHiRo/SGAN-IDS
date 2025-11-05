from __future__ import annotations

from torch.utils.data import Dataset
import torch
import scipy.sparse as sp


class ArrayDataset(Dataset):
    """Simple Dataset wrapping feature matrix (dense or sparse) + optional labels."""

    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        self._is_sparse = sp.issparse(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        feat = self.X[idx]
        if self._is_sparse:
            feat = feat.toarray().squeeze()
        feat = torch.as_tensor(feat, dtype=torch.float32)
        if self.y is None:
            return feat
        return feat, torch.as_tensor(self.y[idx], dtype=torch.long) 