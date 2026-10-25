from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger("sgan_ids")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds third-party code that uses the legacy global RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def environment() -> dict:
    """Library versions, device and git commit, saved with every run (results can depend
    on them: e.g. older scikit-learn MLPs do not accept sample weights)."""
    import importlib.metadata
    import platform
    import subprocess

    info = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy", "torch", "imbalanced-learn", "ctgan"):
        try:
            info[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            info[pkg] = None
    info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = None
    return info


def setup_logging(level: int = logging.INFO) -> None:
    # Third-party libraries (e.g. CTGAN's preprocessing) log at INFO on every column;
    # keep them at WARNING and only show this project's INFO messages.
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    logger.setLevel(level)
