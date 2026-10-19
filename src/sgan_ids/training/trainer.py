"""WGAN-GP training loop.

Design choices (see docs/DECISIONS.md for the reasoning):

- Every critic step draws a *fresh* real batch. Real and fake batches share the
  same labels, so the critic compares p(x | y) with p_G(x | y) directly.
- Classes are drawn with probability proportional to n_c ** alpha. alpha=0.5
  shows rare classes far more often than their frequency without replaying a
  130-row class as often as a 40,000-row class.
- Model selection uses the validation set: every ``eval_every`` steps the
  (EMA) generator is scored by class-averaged MMD to real validation rows, and
  the best checkpoint is kept. WGAN losses are not used for selection because
  the critic estimate drifts as the critic changes and does not track sample
  quality across checkpoints.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sgan_ids.config import GANConfig
from sgan_ids.data.prepare import PreparedData
from sgan_ids.data.transform import Span
from sgan_ids.evaluation.fidelity import median_gamma, mmd_rbf
from sgan_ids.models.networks import Critic, Generator, gradient_penalty
from sgan_ids.utils import logger, resolve_device, set_seed

# Classes with fewer validation rows than this are left out of checkpoint selection:
# an MMD estimated from a handful of rows is mostly noise, and with class averaging
# it would weigh as much as the estimate for a class with thousands of rows.
MIN_VAL_ROWS = 20


class ClassBalancedSampler:
    """Draws labels from p(c) ~ n_c**alpha and real rows of those labels, on device."""

    def __init__(self, y: np.ndarray, n_classes: int, alpha: float, device: torch.device):
        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        weights = np.where(counts > 0, counts**alpha, 0.0)
        self.probs = torch.as_tensor(weights / weights.sum(), dtype=torch.float32, device=device)
        order = np.argsort(y, kind="stable")
        self.order = torch.as_tensor(order, device=device)
        self.starts = torch.as_tensor(
            np.r_[0, np.cumsum(counts)[:-1]], device=device, dtype=torch.long
        )
        self.counts = torch.as_tensor(counts, device=device, dtype=torch.long)

    def labels(self, n: int) -> torch.Tensor:
        return torch.multinomial(self.probs, n, replacement=True)

    def rows(self, labels: torch.Tensor) -> torch.Tensor:
        offset = (torch.rand(len(labels), device=labels.device) * self.counts[labels]).long()
        return self.order[self.starts[labels] + offset]


@dataclass
class TrainResult:
    generator: Generator
    history: pd.DataFrame
    best_step: int
    best_score: float


class ValidationScorer:
    """Class-averaged MMD between generated and real validation rows (lower is better)."""

    def __init__(
        self, X_val: np.ndarray, y_val: np.ndarray, n_classes: int, n_per_class: int, seed: int
    ):
        rng = np.random.default_rng(seed)
        counts = np.bincount(y_val, minlength=n_classes)
        min_rows = MIN_VAL_ROWS if (counts >= MIN_VAL_ROWS).any() else 2
        self.reference = {}
        for c in range(n_classes):
            rows = X_val[y_val == c]
            if len(rows) >= min_rows:
                if len(rows) > n_per_class:
                    rows = rows[rng.choice(len(rows), n_per_class, replace=False)]
                self.reference[c] = rows
        self.excluded = [c for c in range(n_classes) if c not in self.reference]
        self.seed = seed
        # One fixed bandwidth, so scores are comparable across checkpoints.
        self.gamma = median_gamma(np.vstack(list(self.reference.values())), seed=seed)
        self.n_per_class = n_per_class

    def __call__(self, generator: Generator, device: torch.device) -> float:
        # Same latent and Gumbel noise at every evaluation (common random numbers), so the
        # difference between two checkpoints reflects the weights, not the sampling draw.
        # fork_rng keeps the training RNG stream untouched.
        devices = [device.index or 0] if device.type == "cuda" else []
        scores = []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            for c, real in self.reference.items():
                y = torch.full((self.n_per_class,), c, device=device, dtype=torch.long)
                fake = generator.sample(y).cpu().numpy()
                scores.append(mmd_rbf(real, fake, self.gamma))
        return float(np.mean(scores))


def build_models(
    cfg: GANConfig, n_features: int, n_classes: int, spans: list[Span]
) -> tuple[Generator, Critic]:
    generator = Generator(
        cfg.latent_dim, n_classes, spans, cfg.generator_hidden, cfg.class_embed_dim, cfg.gumbel_tau
    )
    critic = Critic(n_features, n_classes, cfg.critic_hidden, cfg.conditioning, cfg.critic_dropout)
    return generator, critic


def train_gan(
    data: PreparedData, cfg: GANConfig, seed: int = 0, out_dir: Path | None = None
) -> TrainResult:
    set_seed(seed)
    device = resolve_device(cfg.device)
    spans = data.transformer.spans
    n_classes = data.n_classes

    X = torch.as_tensor(data.X_train, device=device)
    sampler = ClassBalancedSampler(data.y_train, n_classes, cfg.class_sampling_alpha, device)
    scorer = ValidationScorer(data.X_val, data.y_val, n_classes, cfg.eval_samples_per_class, seed)
    if scorer.excluded:
        logger.info(
            "Checkpoint selection ignores classes with < %d validation rows: %s",
            MIN_VAL_ROWS,
            [data.classes[c] for c in scorer.excluded],
        )

    G, D = build_models(cfg, X.shape[1], n_classes, spans)
    G, D = G.to(device), D.to(device)
    G_ema = copy.deepcopy(G).requires_grad_(False) if cfg.ema_decay > 0 else G
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=tuple(cfg.betas))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=tuple(cfg.betas))

    history, window = [], []
    best_score, best_step, best_state = float("inf"), 0, None
    bs = cfg.batch_size
    logger.info(
        "Training WGAN-GP on %s (%d rows, %d features, %d classes)",
        device,
        len(X),
        X.shape[1],
        n_classes,
    )

    for step in range(1, cfg.n_steps + 1):
        # ---- critic -----------------------------------------------------
        for _ in range(cfg.n_critic):
            y = sampler.labels(bs)
            real = X[sampler.rows(y)]
            with torch.no_grad():
                fake = G(torch.randn(bs, cfg.latent_dim, device=device), y)
            d_real, d_fake = D(real, y).mean(), D(fake, y).mean()
            gp = gradient_penalty(D, real, fake, y)
            loss_d = d_fake - d_real + cfg.gp_lambda * gp
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

        # ---- generator --------------------------------------------------
        y = sampler.labels(bs)
        fake = G(torch.randn(bs, cfg.latent_dim, device=device), y)
        loss_g = -D(fake, y).mean()
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        if G_ema is not G:
            # Warm-up: early on, the average would be dominated by the random initialisation.
            decay = min(cfg.ema_decay, (1 + step) / (10 + step))
            with torch.no_grad():
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.lerp_(p, 1.0 - decay)

        window.append(
            ((d_real - d_fake).detach().item(), gp.detach().item(), loss_g.detach().item())
        )

        # ---- validation & checkpoint selection --------------------------
        if step % cfg.eval_every == 0 or step == cfg.n_steps:
            score = scorer(G_ema, device)
            w, g, lg = np.mean(window, axis=0)
            window = []
            history.append(
                {
                    "step": step,
                    "wasserstein_estimate": w,
                    "gradient_penalty": g,
                    "loss_g": lg,
                    "val_mmd": score,
                }
            )
            improved = score < best_score
            if improved:
                best_score, best_step = score, step
                best_state = copy.deepcopy(G_ema.state_dict())
            logger.info(
                "step %6d | W-est %.4f | GP %.4f | val MMD %.5f%s",
                step,
                w,
                g,
                score,
                "  *" if improved else "",
            )

    G_ema.load_state_dict(best_state)
    G_ema.eval()
    history = pd.DataFrame(history)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_generator(G_ema, cfg, out_dir / "generator.pt", best_step, best_score)
        history.to_csv(out_dir / "history.csv", index=False)
    logger.info("Best checkpoint: step %d (val MMD %.5f)", best_step, best_score)
    return TrainResult(G_ema, history, best_step, best_score)


def save_generator(G: Generator, cfg: GANConfig, path: Path, step: int, score: float) -> None:
    torch.save(
        {
            "state_dict": G.state_dict(),
            "config": asdict(cfg),
            "n_classes": G.n_classes,
            "spans": [asdict(s) for s in G.spans],
            "best_step": step,
            "best_val_mmd": score,
        },
        path,
    )


def load_generator(path: Path, device: torch.device | str = "cpu") -> Generator:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = GANConfig(**{**ckpt["config"], "betas": tuple(ckpt["config"]["betas"])})
    spans = [Span(**s) for s in ckpt["spans"]]
    G = Generator(
        cfg.latent_dim,
        ckpt["n_classes"],
        spans,
        cfg.generator_hidden,
        cfg.class_embed_dim,
        cfg.gumbel_tau,
    )
    G.load_state_dict(ckpt["state_dict"])
    return G.to(device).eval()
