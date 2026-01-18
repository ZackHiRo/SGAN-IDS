"""StealthGAN-IDS Training Module.

Implements WGAN-GP training with:
- Proper train/val/test splits
- Validation-based early stopping
- EMA (Exponential Moving Average) for generator
- Mixed-precision training (AMP)
- Gradient monitoring and logging
- Reproducibility via seeding
"""

import argparse
import csv
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import DataForge, ArrayDataset
from models.generator import Generator
from models.discriminator import Discriminator
from utils.metrics import gradient_penalty
from utils.config import GANConfig
from utils.logging import ExperimentLogger


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMA:
    """Exponential Moving Average for model parameters.
    
    Maintains a shadow copy of model weights that are updated as:
        shadow = decay * shadow + (1 - decay) * current
    
    This provides a smoother, more stable version of the model for evaluation.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow weights with current model weights."""
        for shadow_param, model_param in zip(self.shadow.parameters(), model.parameters()):
            shadow_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
    
    def forward(self, *args, **kwargs):
        """Forward pass through shadow model."""
        return self.shadow(*args, **kwargs)
    
    def state_dict(self):
        return self.shadow.state_dict()
    
    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


def compute_validation_metrics(
    G: nn.Module,
    D: nn.Module,
    val_loader: DataLoader,
    cfg: GANConfig,
    device: torch.device,
    class_weights: torch.Tensor,
) -> Dict[str, float]:
    """Compute validation metrics without gradient computation.
    
    Returns dict with:
    - val_loss_d: Discriminator loss on validation set
    - val_loss_g: Generator loss on validation set
    - val_aux_acc: Auxiliary classifier accuracy
    - val_critic_real: Mean critic score on real samples
    - val_critic_fake: Mean critic score on fake samples
    """
    G.eval()
    D.eval()
    
    ce_loss = nn.CrossEntropyLoss()
    
    total_loss_d = 0.0
    total_loss_g = 0.0
    total_aux_correct = 0
    total_samples = 0
    total_critic_real = 0.0
    total_critic_fake = 0.0
    
    with torch.no_grad():
        for real, labels in val_loader:
            real, labels = real.to(device), labels.to(device)
            bsz = real.size(0)
            
            # Generate fake samples
            z = torch.randn(bsz, cfg.latent_dim, device=device)
            gen_labels = torch.multinomial(class_weights, bsz, replacement=True).to(device)
            fake = G(z, gen_labels)
            
            # Discriminator forward
            d_real, aux_real = D(real, labels)
            d_fake, aux_fake = D(fake, gen_labels)
            
            # Losses
            loss_d = -(d_real.mean() - d_fake.mean())
            loss_g = -d_fake.mean() + ce_loss(aux_fake, gen_labels)
            
            # Auxiliary accuracy
            if aux_real is not None:
                pred = aux_real.argmax(dim=1)
                total_aux_correct += (pred == labels).sum().item()
            
            total_loss_d += loss_d.item() * bsz
            total_loss_g += loss_g.item() * bsz
            total_critic_real += d_real.mean().item() * bsz
            total_critic_fake += d_fake.mean().item() * bsz
            total_samples += bsz
    
    G.train()
    D.train()
    
    return {
        "val_loss_d": total_loss_d / total_samples,
        "val_loss_g": total_loss_g / total_samples,
        "val_aux_acc": total_aux_correct / total_samples if total_samples > 0 else 0.0,
        "val_critic_real": total_critic_real / total_samples,
        "val_critic_fake": total_critic_fake / total_samples,
    }


def train(args):
    """Main training function with all improvements."""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    cfg = GANConfig()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[train] Using device: {device}")
    
    # Enable TF32 for faster training on Ampere+ GPUs
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---------------- Data Loading ----------------
    if args.dataset == "unified":
        datasets = ["nsl_kdd", "cic_ids2017"]
    else:
        datasets = [args.dataset]
    
    forge = DataForge(args.data_root).load(datasets)
    data_split, column_transformer = forge.preprocess(
        val_size=0.15,
        test_size=0.15,
        random_state=args.seed,
    )
    
    data_dim = data_split.n_features
    n_classes = data_split.n_classes
    
    print(f"[train] Data dimensions: {data_dim} features, {n_classes} classes")
    print(f"[train] Train: {len(data_split.X_train)}, Val: {len(data_split.X_val)}, Test: {len(data_split.X_test)}")

    # Create datasets and loaders
    train_dataset = ArrayDataset(data_split.X_train, data_split.y_train)
    val_dataset = ArrayDataset(data_split.X_val, data_split.y_val)
    
    # Weighted sampler for training
    class_weights = torch.from_numpy(data_split.class_weights).float()
    sample_weights = class_weights[data_split.y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    prob = class_weights / class_weights.sum()

    # -------------- Models ----------------
    G = Generator(
        latent_dim=cfg.latent_dim,
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
    
    # EMA for generator (use for evaluation/generation)
    G_ema = EMA(G, decay=cfg.ema_decay if hasattr(cfg, 'ema_decay') else 0.999)
    
    # Optimizers
    optimizer_g = torch.optim.AdamW(G.parameters(), lr=cfg.lr_g, betas=(0.5, 0.9), weight_decay=1e-4)
    optimizer_d = torch.optim.AdamW(D.parameters(), lr=cfg.lr_d, betas=(0.5, 0.9), weight_decay=1e-4)
    
    # LR schedulers (based on validation metrics)
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode='min', patience=15, factor=0.5, min_lr=1e-7
    )
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_d, mode='min', patience=15, factor=0.5, min_lr=1e-7
    )
    
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)  # Built-in label smoothing
    
    # Mixed precision training
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    
    gp_lambda = cfg.gp_lambda

    # --- Resume from checkpoint ---
    start_epoch = 0
    best_val_metric = float('inf')
    
    if args.resume is not None:
        print(f"[train] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        G.load_state_dict(checkpoint["generator_state_dict"])
        D.load_state_dict(checkpoint["discriminator_state_dict"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        if "scheduler_g_state_dict" in checkpoint:
            scheduler_g.load_state_dict(checkpoint["scheduler_g_state_dict"])
        if "scheduler_d_state_dict" in checkpoint:
            scheduler_d.load_state_dict(checkpoint["scheduler_d_state_dict"])
        if "ema_state_dict" in checkpoint:
            G_ema.load_state_dict(checkpoint["ema_state_dict"])
        if "best_val_metric" in checkpoint:
            best_val_metric = checkpoint["best_val_metric"]
        start_epoch = checkpoint["epoch"]
        print(f"[train] Resumed at epoch {start_epoch}")

    # --- Training Loop ---
    with ExperimentLogger() as logger:
        logger.log_params(vars(cfg))
        
        patience_counter = 0
        
        for epoch in range(start_epoch, args.epochs):
            G.train()
            D.train()
            
            epoch_loss_d = 0.0
            epoch_loss_g = 0.0
            epoch_steps = 0
            
            for batch_idx, (real, labels) in enumerate(train_loader):
                real, labels = real.to(device), labels.to(device)
                bsz = real.size(0)
                
                # --------- Critic Updates ----------
                for _ in range(cfg.critic_updates):
                    gen_labels = torch.multinomial(prob, bsz, replacement=True).to(device)
                    z = torch.randn(bsz, cfg.latent_dim, device=device)
                    
                    optimizer_d.zero_grad(set_to_none=True)
                    
                    with autocast(enabled=use_amp):
                        fake = G(z, gen_labels)
                        d_real, aux_real = D(real, labels)
                        d_fake, _ = D(fake.detach(), gen_labels)
                        
                        loss_d_wgan = -(d_real.mean() - d_fake.mean())
                        aux_loss = ce_loss(aux_real, labels) if aux_real is not None else 0.0
                    
                    # Gradient penalty (computed outside autocast for stability)
                    gp = gradient_penalty(D, real, fake.detach(), gp_lambda)
                    loss_d = loss_d_wgan + gp + aux_loss
                    
                    scaler.scale(loss_d).backward()
                    scaler.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=10.0)
                    scaler.step(optimizer_d)
                    scaler.update()
                
                # ------------ Generator Update ------------
                gen_labels = torch.multinomial(prob, bsz, replacement=True).to(device)
                z = torch.randn(bsz, cfg.latent_dim, device=device)
                
                optimizer_g.zero_grad(set_to_none=True)
                
                with autocast(enabled=use_amp):
                    fake = G(z, gen_labels)
                    d_fake, aux_fake, fake_features = D(fake, gen_labels, return_features=True)
                    
                    # Feature matching loss
                    with torch.no_grad():
                        _, _, real_features = D(real, labels, return_features=True)
                    fm_loss = torch.mean((real_features.mean(0) - fake_features.mean(0)) ** 2)
                    
                    loss_g = (
                        -d_fake.mean()
                        + ce_loss(aux_fake, gen_labels)
                        + cfg.feature_matching_weight * fm_loss
                    )
                
                scaler.scale(loss_g).backward()
                scaler.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=10.0)
                scaler.step(optimizer_g)
                scaler.update()
                
                # Update EMA
                G_ema.update(G)
                
                epoch_loss_d += loss_d.item()
                epoch_loss_g += loss_g.item()
                epoch_steps += 1
            
            # --- End of Epoch: Validation ---
            avg_loss_d = epoch_loss_d / epoch_steps
            avg_loss_g = epoch_loss_g / epoch_steps
            
            val_metrics = compute_validation_metrics(
                G, D, val_loader, cfg, device, prob
            )
            
            # Log metrics
            all_metrics = {
                "train_loss_d": avg_loss_d,
                "train_loss_g": avg_loss_g,
                **val_metrics,
            }
            logger.log_metrics(all_metrics, step=epoch)
            
            print(f"Epoch {epoch}: "
                  f"D={avg_loss_d:.4f} G={avg_loss_g:.4f} | "
                  f"Val D={val_metrics['val_loss_d']:.4f} G={val_metrics['val_loss_g']:.4f} "
                  f"Acc={val_metrics['val_aux_acc']:.3f}")
            
            # --- Sample inspection ---
            with torch.no_grad():
                z = torch.randn(cfg.batch_size, cfg.latent_dim, device=device)
                rand_labels = torch.randint(0, n_classes, (cfg.batch_size,), device=device)
                fake_samples = G_ema.forward(z, rand_labels).cpu().numpy()
                print(f"[EMA samples] mean: {fake_samples.mean():.4f}, std: {fake_samples.std():.4f}")
            
            # --- Save training stats ---
            stats_file = "training_stats.csv"
            header = ["epoch", "train_loss_d", "train_loss_g", "val_loss_d", "val_loss_g", 
                      "val_aux_acc", "lr_g", "lr_d"]
            row = [epoch, avg_loss_d, avg_loss_g, val_metrics['val_loss_d'], 
                   val_metrics['val_loss_g'], val_metrics['val_aux_acc'],
                   optimizer_g.param_groups[0]['lr'], optimizer_d.param_groups[0]['lr']]
            write_header = (epoch == start_epoch and not os.path.exists(stats_file))
            with open(stats_file, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                writer.writerow(row)
            
            # --- Early stopping based on VALIDATION metric ---
            val_metric = val_metrics['val_loss_g']  # Use validation generator loss
            
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                patience_counter = 0
                # Save best models (both regular and EMA)
                torch.save(G.state_dict(), "generator_best.pth")
                torch.save(G_ema.state_dict(), "generator_ema_best.pth")
                print(f"[train] New best model saved (val_loss_g={val_metric:.4f})")
            else:
                patience_counter += 1
            
            if patience_counter >= cfg.early_stopping_patience:
                print(f"[train] Early stopping at epoch {epoch} (no improvement for {cfg.early_stopping_patience} epochs)")
                break
            
            # --- LR scheduler step (based on validation) ---
            scheduler_g.step(val_metrics['val_loss_g'])
            scheduler_d.step(val_metrics['val_loss_d'])
            
            # --- Periodic checkpointing ---
            if (epoch + 1) % args.checkpoint_interval == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "generator_state_dict": G.state_dict(),
                    "discriminator_state_dict": D.state_dict(),
                    "ema_state_dict": G_ema.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                    "scheduler_g_state_dict": scheduler_g.state_dict(),
                    "scheduler_d_state_dict": scheduler_d.state_dict(),
                    "best_val_metric": best_val_metric,
                    "config": vars(cfg),
                }
                torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")
                print(f"[train] Checkpoint saved at epoch {epoch+1}")
        
        # --- Save final models ---
        torch.save(G.state_dict(), "generator.pth")
        torch.save(G_ema.state_dict(), "generator_ema.pth")
        torch.save(D.state_dict(), "discriminator.pth")
        print("[train] Final models saved")


def cli():
    p = argparse.ArgumentParser(description="Train StealthGAN-IDS")
    p.add_argument("--data-root", type=Path, default=Path.cwd(), help="Path to datasets")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--cpu", action="store_true", help="Force CPU training")
    p.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N epochs")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--dataset", type=str, default="unified",
                   choices=["nsl_kdd", "cic_ids2017", "cic_ids2018", "unsw_nb15", "unified"],
                   help="Dataset to train on")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--amp", action="store_true", help="Enable mixed-precision training (AMP)")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    train(p.parse_args())


if __name__ == "__main__":
    cli() 