from dataclasses import dataclass


@dataclass
class GANConfig:
    latent_dim: int = 100
    n_classes: int = 2  # benign + attack (expand as needed)
    batch_size: int = 256
    epochs: int = 100
    # --- WGAN-GP stability tuning ---
    lr_g: float = 5e-5  # Lowered learning rate for generator
    lr_d: float = 5e-5  # Lowered learning rate for discriminator (was lr_c)
    critic_updates: int = 3  # Fewer critic updates per generator step
    gp_lambda: float = 20.0  # Increased gradient penalty
    # --- Early stopping and label smoothing ---
    early_stopping_patience: int = 170  # Stop if no improvement after this many epochs
    label_smoothing_real: float = 0.9  # Real label smoothing
    label_smoothing_fake: float = 0.1  # Fake label smoothing
    noisy_label_prob: float = 0.05     # Probability to flip real/fake labels
    # --- Feature matching ---
    feature_matching_weight: float = 2.0  # Strength of feature matching loss 