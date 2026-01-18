"""Configuration management for StealthGAN-IDS.

Supports:
- Dataclass-based configuration with type hints
- YAML/JSON loading and saving
- Validation of configuration values
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple, Optional, Any, Dict
import json

__all__ = ["GANConfig", "EvalConfig", "load_config", "save_config"]


@dataclass
class GANConfig:
    """Training configuration for StealthGAN.
    
    Attributes organized by category:
    - Model architecture
    - Training hyperparameters
    - WGAN-GP stability
    - Regularization
    - Early stopping
    """
    
    # --- Model Architecture ---
    latent_dim: int = 100
    generator_hidden_dims: Tuple[int, ...] = (256, 512, 512, 256)
    generator_n_residual: int = 2
    discriminator_hidden_dims: Tuple[int, ...] = (256, 128, 64)
    use_spectral_norm: bool = True
    use_attention: bool = True
    
    # --- Training ---
    batch_size: int = 256
    epochs: int = 100
    lr_g: float = 2e-4  # Generator learning rate
    lr_d: float = 2e-4  # Discriminator learning rate
    betas: Tuple[float, float] = (0.5, 0.9)  # Adam betas
    weight_decay: float = 1e-4
    
    # --- WGAN-GP Stability ---
    critic_updates: int = 5  # Discriminator updates per generator update
    gp_lambda: float = 10.0  # Gradient penalty coefficient
    
    # --- Regularization ---
    dropout: float = 0.1
    label_smoothing: float = 0.1  # For auxiliary classifier
    feature_matching_weight: float = 1.0  # Feature matching loss weight
    
    # --- EMA ---
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # --- Early Stopping ---
    early_stopping_patience: int = 30  # Epochs without improvement
    min_lr: float = 1e-7  # Minimum learning rate
    lr_patience: int = 15  # Epochs before LR reduction
    lr_factor: float = 0.5  # LR reduction factor
    
    # --- Deprecated (kept for backward compatibility) ---
    label_smoothing_real: float = 0.9
    label_smoothing_fake: float = 0.1
    noisy_label_prob: float = 0.05
    
    def __post_init__(self):
        """Validate configuration values."""
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert 0 < self.lr_g < 1, "lr_g must be in (0, 1)"
        assert 0 < self.lr_d < 1, "lr_d must be in (0, 1)"
        assert self.gp_lambda >= 0, "gp_lambda must be non-negative"
        assert 0 <= self.ema_decay < 1, "ema_decay must be in [0, 1)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EvalConfig:
    """Evaluation configuration.
    
    Controls generation, visualization, and downstream evaluation.
    """
    
    # --- Generation ---
    n_synthetic_per_class: int = 2000
    target_minority: bool = True
    minority_threshold: float = 0.01  # Classes below this fraction are "minority"
    
    # --- Visualization ---
    tsne_perplexity: int = 30
    tsne_n_samples: int = 2000
    umap_n_neighbors: int = 15
    plot_dpi: int = 150
    
    # --- Downstream Evaluation ---
    cv_folds: int = 5
    classifiers: Tuple[str, ...] = ("random_forest", "xgboost", "lightgbm", "mlp")
    n_estimators: int = 100
    
    # --- Quality Metrics ---
    compute_mmd: bool = True
    compute_precision_recall: bool = True
    knn_k: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path, config_class: type = GANConfig) -> GANConfig | EvalConfig:
    """Load configuration from YAML or JSON file.
    
    Args:
        path: Path to configuration file
        config_class: Configuration class to instantiate
        
    Returns:
        Configuration object
    """
    path = Path(path)
    
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML config. Install with: pip install pyyaml")
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    
    # Convert lists to tuples for tuple fields
    for field_name, field_type in config_class.__dataclass_fields__.items():
        if field_name in data and "Tuple" in str(field_type.type):
            data[field_name] = tuple(data[field_name])
    
    return config_class(**data)


def save_config(config: GANConfig | EvalConfig, path: str | Path) -> None:
    """Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration object
        path: Output path (format determined by extension)
    """
    path = Path(path)
    data = config.to_dict()
    
    # Convert tuples to lists for serialization
    for key, value in data.items():
        if isinstance(value, tuple):
            data[key] = list(value)
    
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            raise ImportError("PyYAML required for YAML config. Install with: pip install pyyaml")
    elif path.suffix == ".json":
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}") 