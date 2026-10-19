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


def setup_logging(level: int = logging.INFO) -> None:
    # Third-party libraries (e.g. CTGAN's preprocessing) log at INFO on every column;
    # keep them at WARNING and only show this project's INFO messages.
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    logger.setLevel(level)
