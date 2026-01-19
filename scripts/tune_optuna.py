#!/usr/bin/env python
"""Optuna-based hyperparameter tuning for StealthGAN-IDS.

Tunes hyperparameters against downstream IDS classifier performance,
not raw GAN losses. Uses early pruning to save compute on bad trials.

Usage:
    python scripts/tune_optuna.py --data-root /path/to/data --dataset cic_ids2018 \
        --n-trials 50 --tune-epochs 15 --max-samples 100000

The best hyperparameters are saved to `best_hyperparams.json` and can be used
for a full training run.
"""

import sys
import pathlib

project_root = pathlib.Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import json
import gc
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    print("Optuna not installed. Run: pip install optuna")
    sys.exit(1)

from data import DataForge, ArrayDataset
from models.generator import Generator
from models.discriminator import Discriminator
from utils.metrics import gradient_penalty


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        for shadow_param, model_param in zip(self.shadow.parameters(), model.parameters()):
            shadow_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
    
    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)
    
    def state_dict(self):
        return self.shadow.state_dict()


def generate_synthetic_samples(
    G: nn.Module,
    n_classes: int,
    latent_dim: int,
    device: torch.device,
    n_per_class: int = 500,
    batch_size: int = 256,
) -> tuple:
    """Generate synthetic samples for evaluation."""
    G.eval()
    all_samples = []
    all_labels = []
    
    with torch.no_grad():
        for cls in range(n_classes):
            n_gen = n_per_class
            while n_gen > 0:
                bsz = min(batch_size, n_gen)
                z = torch.randn(bsz, latent_dim, device=device)
                labels = torch.full((bsz,), cls, dtype=torch.long, device=device)
                fake = G(z, labels).cpu().numpy()
                all_samples.append(fake)
                all_labels.append(labels.cpu().numpy())
                n_gen -= bsz
    
    return np.vstack(all_samples), np.hstack(all_labels)


def compute_distribution_quality(X_real: np.ndarray, X_synth: np.ndarray) -> Dict[str, float]:
    """Compute distribution quality metrics between real and synthetic data.
    
    Returns metrics that don't require labels - useful when F1 is already saturated.
    """
    # Feature-wise statistics comparison
    real_mean = X_real.mean(axis=0)
    synth_mean = X_synth.mean(axis=0)
    real_std = X_real.std(axis=0) + 1e-8
    synth_std = X_synth.std(axis=0) + 1e-8
    
    # Mean absolute error of feature means (normalized)
    mean_mae = np.abs(real_mean - synth_mean).mean()
    
    # Mean absolute error of feature stds (normalized) 
    std_mae = np.abs(real_std - synth_std).mean()
    
    # Coverage: what fraction of real feature ranges are covered by synthetic
    real_min, real_max = X_real.min(axis=0), X_real.max(axis=0)
    synth_min, synth_max = X_synth.min(axis=0), X_synth.max(axis=0)
    real_range = real_max - real_min + 1e-8
    
    # Overlap ratio per feature
    overlap_min = np.maximum(real_min, synth_min)
    overlap_max = np.minimum(real_max, synth_max)
    overlap = np.maximum(0, overlap_max - overlap_min)
    coverage = (overlap / real_range).mean()
    
    # Diversity: ratio of synthetic std to real std (want close to 1)
    diversity_ratio = (synth_std / real_std).mean()
    
    return {
        "mean_mae": mean_mae,
        "std_mae": std_mae,
        "coverage": coverage,
        "diversity_ratio": diversity_ratio,
    }


def quick_evaluate(
    G: nn.Module,
    X_real: np.ndarray,
    y_real: np.ndarray,
    n_classes: int,
    latent_dim: int,
    device: torch.device,
    n_synth_per_class: int = 400,
) -> Dict[str, float]:
    """Quick evaluation for hyperparameter tuning.
    
    Uses a multi-signal objective:
    1. F1 improvement (if baseline isn't already perfect)
    2. Distribution quality (always meaningful)
    3. Per-class balance (smaller classes get more weight)
    
    This handles cases where:
    - No true "minority" classes exist (all balanced)
    - Baseline F1 is already near-perfect (saturated)
    """
    # Generate synthetic samples
    X_synth, y_synth = generate_synthetic_samples(
        G, n_classes, latent_dim, device,
        n_per_class=n_synth_per_class,
        batch_size=256,
    )
    
    # Split real data for baseline vs augmented comparison (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real, test_size=0.3, random_state=42, stratify=y_real
    )
    
    # --- Identify "smaller" classes (below-average representation) ---
    # This is more useful than absolute threshold when classes are balanced
    class_counts = np.bincount(y_test, minlength=n_classes)
    total = class_counts.sum()
    avg_share = 1.0 / n_classes
    # Classes with less than 80% of equal share are "smaller"
    smaller_classes = np.where((class_counts / total) < (avg_share * 0.8))[0]
    
    # --- Train classifiers ---
    # Baseline: train only on real data
    clf_baseline = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    clf_baseline.fit(X_train, y_train)
    y_pred_baseline = clf_baseline.predict(X_test)
    
    # Augmented: train on real + synthetic
    X_aug = np.vstack([X_train, X_synth])
    y_aug = np.hstack([y_train, y_synth])
    clf_aug = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    clf_aug.fit(X_aug, y_aug)
    y_pred_aug = clf_aug.predict(X_test)
    
    # --- Overall macro F1 ---
    f1_baseline = f1_score(y_test, y_pred_baseline, average="macro", zero_division=0)
    f1_aug = f1_score(y_test, y_pred_aug, average="macro", zero_division=0)
    
    # --- Per-class F1 ---
    per_class_f1_baseline = f1_score(y_test, y_pred_baseline, average=None, zero_division=0)
    per_class_f1_aug = f1_score(y_test, y_pred_aug, average=None, zero_division=0)
    
    # --- Weighted improvement: weight classes inversely by their size ---
    # Smaller classes get more weight in the objective
    class_weights = 1.0 / (class_counts / total + 0.01)  # Inverse frequency
    class_weights = class_weights / class_weights.sum()  # Normalize
    
    per_class_improvement = per_class_f1_aug - per_class_f1_baseline
    weighted_improvement = (per_class_improvement * class_weights).sum()
    
    # --- Smaller-class F1 (if any exist) ---
    if len(smaller_classes) > 0:
        f1_smaller_baseline = f1_score(
            y_test, y_pred_baseline,
            labels=smaller_classes, average="macro", zero_division=0
        )
        f1_smaller_aug = f1_score(
            y_test, y_pred_aug,
            labels=smaller_classes, average="macro", zero_division=0
        )
        smaller_improvement = f1_smaller_aug - f1_smaller_baseline
    else:
        f1_smaller_baseline = f1_baseline
        f1_smaller_aug = f1_aug
        smaller_improvement = f1_aug - f1_baseline
    
    # --- Distribution quality metrics ---
    dist_quality = compute_distribution_quality(X_test, X_synth)
    
    # Quality score: higher is better (coverage close to 1, diversity close to 1, low MAE)
    # Normalize to roughly 0-1 scale
    quality_score = (
        0.4 * dist_quality["coverage"] +
        0.3 * min(dist_quality["diversity_ratio"], 1.0) +  # Cap at 1
        0.15 * max(0, 1 - dist_quality["mean_mae"]) +
        0.15 * max(0, 1 - dist_quality["std_mae"])
    )
    
    # --- Collapse detection ---
    synth_std = X_synth.std()
    per_class_std = []
    for cls in range(n_classes):
        mask = y_synth == cls
        if mask.sum() > 0:
            per_class_std.append(X_synth[mask].std())
    min_class_std = min(per_class_std) if per_class_std else 0.0
    
    # --- Headroom: how much room for improvement ---
    headroom = 1.0 - f1_baseline  # If baseline is 0.99, headroom is 0.01
    
    return {
        # Overall metrics
        "baseline_f1": f1_baseline,
        "augmented_f1": f1_aug,
        "improvement": f1_aug - f1_baseline,
        # Weighted improvement (favors smaller classes)
        "weighted_improvement": weighted_improvement,
        # Smaller-class metrics
        "smaller_baseline_f1": f1_smaller_baseline,
        "smaller_augmented_f1": f1_smaller_aug,
        "smaller_improvement": smaller_improvement,
        "n_smaller_classes": len(smaller_classes),
        # Distribution quality
        "quality_score": quality_score,
        "coverage": dist_quality["coverage"],
        "diversity_ratio": dist_quality["diversity_ratio"],
        # Collapse detection
        "synth_std": synth_std,
        "min_class_std": min_class_std,
        # Headroom (for adaptive weighting)
        "headroom": headroom,
        # Diagnostics
        "per_class_f1_baseline": per_class_f1_baseline.tolist(),
        "per_class_f1_aug": per_class_f1_aug.tolist(),
        "per_class_improvement": per_class_improvement.tolist(),
    }


def train_trial(
    trial: optuna.Trial,
    data_split,
    device: torch.device,
    n_epochs: int = 15,
    seed: int = 42,
) -> float:
    """Train a single trial with suggested hyperparameters.
    
    Returns an adaptive objective that combines:
    - Weighted F1 improvement (smaller classes weighted more)
    - Distribution quality (coverage, diversity)
    
    The weighting adapts based on baseline F1:
    - If baseline is near-perfect, focus on distribution quality
    - Otherwise, focus on F1 improvement
    """
    set_seed(seed)
    
    # --- Suggest hyperparameters ---
    lr_g = trial.suggest_float("lr_g", 1e-5, 5e-4, log=True)
    lr_d = trial.suggest_float("lr_d", 1e-5, 5e-4, log=True)
    critic_updates = trial.suggest_int("critic_updates", 1, 5)
    gp_lambda = trial.suggest_float("gp_lambda", 1.0, 20.0)
    feature_matching_weight = trial.suggest_float("feature_matching_weight", 0.1, 5.0)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    latent_dim = trial.suggest_categorical("latent_dim", [64, 100, 128])
    ema_decay = trial.suggest_float("ema_decay", 0.99, 0.9999)
    
    # Log hyperparameters
    print(f"\n[Trial {trial.number}] lr_g={lr_g:.2e}, lr_d={lr_d:.2e}, "
          f"critic={critic_updates}, gp={gp_lambda:.1f}, fm={feature_matching_weight:.2f}, "
          f"batch={batch_size}")
    
    data_dim = data_split.n_features
    n_classes = data_split.n_classes
    
    # Create datasets
    train_dataset = ArrayDataset(data_split.X_train, data_split.y_train)
    
    class_weights = torch.from_numpy(data_split.class_weights).float()
    sample_weights = class_weights[data_split.y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=0,  # Avoid multiprocessing issues in Optuna
        pin_memory=True if device.type == "cuda" else False,
    )
    
    prob = class_weights / class_weights.sum()
    
    # --- Models ---
    G = Generator(
        latent_dim=latent_dim,
        out_dim=data_dim,
        n_classes=n_classes,
        hidden_dims=(256, 512, 512, 256),
        n_residual_blocks=2,
        dropout=0.1,
    ).to(device)
    
    D = Discriminator(
        in_dim=data_dim,
        n_classes=n_classes,
        hidden_dims=(256, 128, 64),
        use_spectral_norm=True,
        dropout=0.1,
    ).to(device)
    
    G_ema = EMA(G, decay=ema_decay)
    
    optimizer_g = torch.optim.AdamW(G.parameters(), lr=lr_g, betas=(0.5, 0.9), weight_decay=1e-4)
    optimizer_d = torch.optim.AdamW(D.parameters(), lr=lr_d, betas=(0.5, 0.9), weight_decay=1e-4)
    
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # --- Training loop ---
    best_improvement = -float('inf')
    
    for epoch in range(n_epochs):
        G.train()
        D.train()
        
        epoch_loss_d = 0.0
        epoch_loss_g = 0.0
        epoch_steps = 0
        
        for batch_idx, (real, labels) in enumerate(train_loader):
            real, labels = real.to(device), labels.to(device)
            bsz = real.size(0)
            
            # --- Critic updates ---
            for _ in range(critic_updates):
                gen_labels = torch.multinomial(prob, bsz, replacement=True).to(device)
                z = torch.randn(bsz, latent_dim, device=device)
                
                optimizer_d.zero_grad(set_to_none=True)
                
                fake = G(z, gen_labels)
                d_real, aux_real = D(real, labels)
                d_fake, _ = D(fake.detach(), gen_labels)
                
                loss_d_wgan = -(d_real.mean() - d_fake.mean())
                aux_loss = ce_loss(aux_real, labels) if aux_real is not None else 0.0
                gp = gradient_penalty(D, real, fake.detach(), gp_lambda)
                loss_d = loss_d_wgan + gp + aux_loss
                
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=10.0)
                optimizer_d.step()
            
            # --- Generator update ---
            gen_labels = torch.multinomial(prob, bsz, replacement=True).to(device)
            z = torch.randn(bsz, latent_dim, device=device)
            
            optimizer_g.zero_grad(set_to_none=True)
            
            fake = G(z, gen_labels)
            d_fake, aux_fake, fake_features = D(fake, gen_labels, return_features=True)
            
            with torch.no_grad():
                _, _, real_features = D(real, labels, return_features=True)
            fm_loss = torch.mean((real_features.mean(0) - fake_features.mean(0)) ** 2)
            
            loss_g = -d_fake.mean() + ce_loss(aux_fake, gen_labels) + feature_matching_weight * fm_loss
            
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=10.0)
            optimizer_g.step()
            
            G_ema.update(G)
            
            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            epoch_steps += 1
        
        avg_loss_d = epoch_loss_d / epoch_steps
        avg_loss_g = epoch_loss_g / epoch_steps
        
        # --- Evaluate every few epochs ---
        if (epoch + 1) % 3 == 0 or epoch == n_epochs - 1:
            eval_results = quick_evaluate(
                G_ema.shadow,
                data_split.X_val,
                data_split.y_val,
                n_classes,
                latent_dim,
                device,
                n_synth_per_class=400,
            )
            
            # Extract metrics
            weighted_imp = eval_results["weighted_improvement"]
            quality = eval_results["quality_score"]
            coverage = eval_results["coverage"]
            headroom = eval_results["headroom"]
            synth_std = eval_results["synth_std"]
            min_class_std = eval_results["min_class_std"]
            
            # --- Adaptive objective ---
            # If baseline is already near-perfect (headroom < 0.05), rely more on quality
            # Otherwise, weight improvement more heavily
            if headroom < 0.05:
                # F1 is saturated, focus on distribution quality
                objective_value = 0.3 * weighted_imp + 0.7 * quality
            else:
                # Room for F1 improvement, balance both
                objective_value = 0.6 * weighted_imp + 0.4 * quality
            
            print(f"  Epoch {epoch}: D={avg_loss_d:.2f} G={avg_loss_g:.2f} | "
                  f"F1={eval_results['augmented_f1']:.3f} (Δ={weighted_imp:+.4f}) "
                  f"Q={quality:.3f} cov={coverage:.2f} "
                  f"obj={objective_value:.4f}")
            
            # Report intermediate value for pruning
            trial.report(objective_value, epoch)
            
            # Check for mode collapse (global or per-class)
            if synth_std < 0.01:
                print(f"  [!] Mode collapse detected (global std={synth_std:.4f}), pruning trial")
                raise optuna.TrialPruned()
            if min_class_std < 0.005:
                print(f"  [!] Per-class collapse detected (min_class_std={min_class_std:.4f}), pruning trial")
                raise optuna.TrialPruned()
            
            # Prune if unpromising
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            if objective_value > best_improvement:
                best_improvement = objective_value
    
    # Cleanup
    del G, D, G_ema, optimizer_g, optimizer_d
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return best_improvement


def main(args):
    print("=" * 60)
    print("StealthGAN-IDS Hyperparameter Tuning with Optuna")
    print("=" * 60)
    
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[tune] Using device: {device}")
    
    # --- Load data ---
    print("\n[1/3] Loading data...")
    if args.dataset == "unified":
        datasets = ["nsl_kdd", "cic_ids2017"]
    else:
        datasets = [args.dataset]
    
    max_samples = args.max_samples
    if max_samples:
        print(f"[tune] Limiting dataset to {max_samples} samples")
    
    forge = DataForge(args.data_root, max_samples=max_samples).load(datasets)
    data_split, _ = forge.preprocess(
        val_size=0.15,
        test_size=0.15,
        random_state=args.seed,
    )
    
    print(f"[tune] Data: {data_split.n_features} features, {data_split.n_classes} classes")
    print(f"[tune] Train: {len(data_split.X_train)}, Val: {len(data_split.X_val)}")
    
    # Show class distribution for context
    class_counts = np.bincount(data_split.y_val, minlength=data_split.n_classes)
    total = class_counts.sum()
    avg_share = 1.0 / data_split.n_classes
    smaller_classes = np.where((class_counts / total) < (avg_share * 0.8))[0]
    print(f"[tune] Smaller classes (<{avg_share*0.8*100:.0f}% of equal share): {len(smaller_classes)} of {data_split.n_classes}")
    for i, count in enumerate(class_counts):
        pct = 100 * count / total
        marker = " [SMALLER]" if i in smaller_classes else ""
        print(f"        Class {i}: {count:6d} ({pct:5.1f}%){marker}")
    
    print(f"\n[tune] Objective = weighted F1 improvement + distribution quality")
    print(f"[tune] (adapts based on baseline F1 headroom)")
    
    # --- Create Optuna study ---
    print("\n[2/3] Creating Optuna study...")
    
    # Use TPE sampler (good for GAN hyperparameters)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    
    # Use median pruner to stop bad trials early
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=3,
        interval_steps=1,
    )
    
    study = optuna.create_study(
        direction="maximize",  # Maximize F1 improvement
        sampler=sampler,
        pruner=pruner,
        study_name="stealthgan_tuning",
    )
    
    # --- Run optimization ---
    print(f"\n[3/3] Running {args.n_trials} trials ({args.tune_epochs} epochs each)...")
    print("=" * 60)
    
    def objective(trial):
        return train_trial(
            trial,
            data_split,
            device,
            n_epochs=args.tune_epochs,
            seed=args.seed,
        )
    
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=True,
    )
    
    # --- Results ---
    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    
    # Show trial statistics
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print(f"\nStudy statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")
    
    # Best trial
    print(f"\nBest trial:")
    trial = study.best_trial
    print(f"  Objective value: {trial.value:.4f}")
    print(f"  (combines weighted F1 improvement + distribution quality)")
    print(f"  Params:")
    for key, value in trial.params.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")
    
    # Save best hyperparameters
    best_params = {
        "best_value": trial.value,
        "params": trial.params,
        "n_trials": len(study.trials),
        "dataset": args.dataset,
    }
    
    output_file = Path(args.output) if args.output else Path("best_hyperparams.json")
    with open(output_file, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\n✅ Best hyperparameters saved to {output_file}")
    
    # Show top 5 trials
    print("\nTop 5 trials:")
    trials_df = study.trials_dataframe()
    if not trials_df.empty:
        trials_df = trials_df.sort_values("value", ascending=False).head(5)
        for idx, row in trials_df.iterrows():
            print(f"  Trial {row['number']}: objective = {row['value']:.4f}")
    
    # Generate config snippet
    print("\n" + "-" * 60)
    print("To use these hyperparameters, update utils/config.py:")
    print("-" * 60)
    print(f"""
@dataclass
class GANConfig:
    # Tuned hyperparameters
    latent_dim: int = {trial.params.get('latent_dim', 100)}
    batch_size: int = {trial.params.get('batch_size', 256)}
    lr_g: float = {trial.params['lr_g']:.6f}
    lr_d: float = {trial.params['lr_d']:.6f}
    critic_updates: int = {trial.params['critic_updates']}
    gp_lambda: float = {trial.params['gp_lambda']:.2f}
    feature_matching_weight: float = {trial.params['feature_matching_weight']:.2f}
    ema_decay: float = {trial.params['ema_decay']:.4f}
""")


def cli():
    p = argparse.ArgumentParser(
        description="Tune StealthGAN-IDS hyperparameters with Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick tuning (fewer trials, shorter epochs)
  python scripts/tune_optuna.py --data-root /data --dataset cic_ids2018 \\
      --n-trials 20 --tune-epochs 10 --max-samples 50000
  
  # Full tuning (more thorough)
  python scripts/tune_optuna.py --data-root /data --dataset cic_ids2018 \\
      --n-trials 100 --tune-epochs 20 --max-samples 100000
  
  # Tuning with more samples for balanced datasets
  python scripts/tune_optuna.py --data-root /data --dataset cic_ids2018 \\
      --max-samples 300000 --n-trials 50
        """
    )
    
    # Data
    p.add_argument("--data-root", type=Path, default=Path.cwd(),
                   help="Path to datasets")
    p.add_argument("--dataset", type=str, default="cic_ids2018",
                   choices=["nsl_kdd", "cic_ids2017", "cic_ids2018", "unsw_nb15", "unified"],
                   help="Dataset to tune on")
    p.add_argument("--max-samples", type=int, default=100000,
                   help="Max samples to use (smaller = faster tuning)")
    
    # Tuning
    p.add_argument("--n-trials", type=int, default=50,
                   help="Number of Optuna trials")
    p.add_argument("--tune-epochs", type=int, default=15,
                   help="Epochs per trial (fewer = faster, less accurate)")
    p.add_argument("--timeout", type=int, default=None,
                   help="Timeout in seconds (optional)")
    
    # Output
    p.add_argument("--output", type=str, default="best_hyperparams.json",
                   help="Output file for best hyperparameters")
    
    # Other
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    
    args = p.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
