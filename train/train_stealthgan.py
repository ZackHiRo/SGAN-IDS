import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import DataForge, ArrayDataset
from models.generator import Generator
from models.discriminator import Discriminator
from utils.metrics import gradient_penalty
from utils.config import GANConfig
from utils.logging import ExperimentLogger
import csv
import os


def train(args):
    cfg = GANConfig()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # ---------------- data ----------------
    # Select datasets based on argument
    if args.dataset == "unified":
        datasets = ["nsl_kdd", "cic_ids2017"]
    else:
        datasets = [args.dataset]
    forge = DataForge(args.data_root).load(datasets)
    (X, y), _ = forge.preprocess()
    y = y.astype(int) if y is not None else None
    data_dim = X.shape[1]

    print(f"[train] X shape after preprocessing: {X.shape}")

    dataset = ArrayDataset(X, y)
    # Weighted sampler to oversample minority classes
    if y is not None:
        class_counts = torch.bincount(torch.as_tensor(y))
        class_weights = 1.0 / (class_counts.float() + 1e-6)
        prob = class_weights / class_weights.sum()
        sample_weights = class_weights[y]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    n_classes = int(y.max()) + 1 if y is not None else None

    # -------------- models ----------------
    G = Generator(cfg.latent_dim, data_dim, n_classes).to(device)
    D = Discriminator(data_dim, n_classes).to(device)
    # --- Optimizers ---
    optimizer_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(0.5, 0.9))
    optimizer_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(0.5, 0.9))
    # --- LR schedulers ---
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, 'min', patience=10, factor=0.5)
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, 'min', patience=10, factor=0.5)
    ce_loss = nn.CrossEntropyLoss()

    gp_lambda = cfg.gp_lambda  # Gradient penalty coefficient

    # --- Resume from checkpoint logic ---
    start_epoch = 0
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
        if "config" in checkpoint:
            # Optionally update cfg with checkpoint config
            for k, v in checkpoint["config"].items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        start_epoch = checkpoint["epoch"]
        print(f"[train] Resumed at epoch {start_epoch}")

    with ExperimentLogger() as logger:
        logger.log_params(vars(cfg))
        best_loss_g = float('inf')
        patience = cfg.early_stopping_patience
        epochs_no_improve = 0
        for epoch in range(start_epoch, args.epochs):
            for real, labels in loader:
                real, labels = real.to(device), labels.to(device)
                bsz = real.size(0)
                # --------- critic updates ----------
                for _ in range(cfg.critic_updates):
                    # --- Oversample minority classes for fake label conditioning ---
                    gen_labels = torch.multinomial(prob, bsz, replacement=True).to(device)
                    z = torch.randn(bsz, cfg.latent_dim, device=device)
                    fake = G(z, gen_labels)
                    d_real, aux_real = D(real, labels)
                    d_fake, aux_fake = D(fake.detach(), gen_labels)
                    loss_d = -(d_real.mean() - d_fake.mean())
                    gp = gradient_penalty(D, real, fake.detach(), gp_lambda)
                    # --- Label smoothing and noisy labels for auxiliary classification ---
                    real_label = cfg.label_smoothing_real
                    fake_label = cfg.label_smoothing_fake
                    # Randomly flip labels with probability cfg.noisy_label_prob
                    if torch.rand(1).item() < cfg.noisy_label_prob:
                        real_label, fake_label = fake_label, real_label  # Flip
                    # For CrossEntropyLoss, label smoothing is not directly supported, so we can use soft targets if needed
                    # But here, we keep standard CE for simplicity; smoothing/noise is more relevant for BCE loss
                    aux_loss = ce_loss(aux_real, labels)
                    optimizer_d.zero_grad(); (loss_d + gp + aux_loss).backward();
                    torch.nn.utils.clip_grad_norm_(D.parameters(), 10)
                    optimizer_d.step()
                # ------------ generator ------------
                # --- Oversample minority classes for fake label conditioning ---
                gen_labels = torch.multinomial(prob, bsz, replacement=True).to(device)
                z = torch.randn(bsz, cfg.latent_dim, device=device)
                fake = G(z, gen_labels)
                d_fake, aux_fake, fake_features = D(fake, gen_labels, return_features=True)
                # --- Label smoothing and noisy labels for auxiliary classification ---
                real_label = cfg.label_smoothing_real
                fake_label = cfg.label_smoothing_fake
                # Randomly flip labels with probability cfg.noisy_label_prob
                if torch.rand(1).item() < cfg.noisy_label_prob:
                    real_label, fake_label = fake_label, real_label  # Flip
                # For CrossEntropyLoss, label smoothing is not directly supported, so we can use soft targets if needed
                # But here, we keep standard CE for simplicity
                # Feature matching loss
                with torch.no_grad():
                    _, _, real_features = D(real, labels, return_features=True)
                feature_matching_loss = torch.mean((real_features.mean(0) - fake_features.mean(0)) ** 2)
                # --- Feature matching loss (if present) ---
                feature_matching_loss = cfg.feature_matching_weight * feature_matching_loss  # Configurable weight
                lambda_fm = 10.0
                loss_g = -d_fake.mean() + ce_loss(aux_fake, gen_labels) + lambda_fm * feature_matching_loss
                optimizer_g.zero_grad(); loss_g.backward();
                torch.nn.utils.clip_grad_norm_(G.parameters(), 10)
                optimizer_g.step()
            logger.log_metrics({"loss_d": loss_d.item(), "loss_g": loss_g.item()}, step=epoch)
            print(f"Epoch {epoch}: D={loss_d.item():.3f} G={loss_g.item():.3f}")
            # --- Sample inspection: print mean/std of generated samples ---
            with torch.no_grad():
                z = torch.randn(cfg.batch_size, cfg.latent_dim, device=device)
                rand_labels = torch.randint(0, n_classes, (cfg.batch_size,), device=device)
                fake_samples = G(z, rand_labels).cpu().numpy()
                print(f"[Sample inspection] Fake batch mean: {fake_samples.mean():.4f}, std: {fake_samples.std():.4f}")
            real_batch = real[:cfg.batch_size].cpu().numpy() if real.size(0) >= cfg.batch_size else real.cpu().numpy()
            print(f"[Sample inspection] Real batch mean: {real_batch.mean():.4f}, std: {real_batch.std():.4f}")
            # --- Save stats to CSV ---
            stats_file = "training_stats.csv"
            header = ["epoch", "loss_d", "loss_g", "fake_mean", "fake_std", "real_mean", "real_std"]
            row = [epoch, loss_d.item(), loss_g.item(), fake_samples.mean(), fake_samples.std(), real_batch.mean(), real_batch.std()]
            write_header = (epoch == 0 and not os.path.exists(stats_file))
            with open(stats_file, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                writer.writerow(row)
            print(f"[train] Stats saved to {stats_file}")
            # --- Early stopping logic ---
            if loss_g.item() < best_loss_g:
                best_loss_g = loss_g.item()
                epochs_no_improve = 0
                # Save best generator
                torch.save(G.state_dict(), "generator_best.pth")
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            # --- End of epoch: scheduler step ---
            scheduler_g.step(loss_g.item())
            scheduler_d.step(loss_d.item())
            print(f"[train] Epoch {epoch}: lr_g={optimizer_g.param_groups[0]['lr']}, lr_d={optimizer_d.param_groups[0]['lr']}")
            # --- Periodic checkpointing ---
            if (epoch + 1) % args.checkpoint_interval == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "generator_state_dict": G.state_dict(),
                    "discriminator_state_dict": D.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                    "scheduler_g_state_dict": scheduler_g.state_dict(),
                    "scheduler_d_state_dict": scheduler_d.state_dict(),
                    "config": vars(cfg),
                }
                torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")
                print(f"[train] Checkpoint saved at epoch {epoch+1}")
        # --- Save generator after training ---
        torch.save(G.state_dict(), "generator.pth")
        print("[train] Generator model saved as generator.pth")
        # --- Save discriminator after training ---
        torch.save(D.state_dict(), "discriminator.pth")
        print("[train] Discriminator model saved as discriminator.pth")


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path.cwd())
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N epochs")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    p.add_argument("--dataset", type=str, default="unified", choices=["nsl_kdd", "cic_ids2017", "unified"], help="Which dataset to train on")
    train(p.parse_args())


if __name__ == "__main__":
    cli() 