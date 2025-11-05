## StealthGAN‑IDS

StealthGAN‑IDS is an end‑to‑end pipeline to synthesize realistic, under‑represented network attacks for IDS augmentation. It implements:

- Data ingestion and preprocessing (DataForge)
- A class‑conditioned WGAN‑GP with auxiliary classification and self‑attention (StealthGAN)
- Training orchestration with checkpointing and experiment logging (ForgeTrain)
- Evaluation and augmentation tests with t‑SNE and IDS uplift (AugmentEval)

Key references: [WGAN‑GP](https://arxiv.org/abs/1704.00028), [AC‑GAN](https://arxiv.org/abs/1610.09585), [t‑SNE](https://jmlr.org/papers/v9/vandermaaten08a.html), [MLflow](https://mlflow.org/).

### Folder structure
```text
stealthgan_ids/
├── data/                  # DataForge: loading, cleaning, encoding
├── models/                # Generator, Discriminator, Self‑Attention
├── train/                 # ForgeTrain: training loop & CLI
├── eval/                  # AugmentEval: metrics, plots, IDS tests
├── utils/                 # Metrics, config, MLflow logging
├── scripts/               # CLI entry points (preprocess/train/eval)
├── mlruns/                # Local MLflow runs (if mlflow installed)
└── eval_outputs/          # Generated plots & reports
```

## 1) Installation

Requirements:

- Python 3.9+ (recommended)
- PyTorch (CPU or CUDA). Select the correct wheel from [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- Optional: [MLflow](https://mlflow.org/) for experiment tracking

Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt

# (Optional) install MLflow for experiment logging
pip install mlflow
```

CUDA users: install the matching torch/torchvision with the appropriate CUDA version from the official instructions linked above.

## 2) Datasets and Preprocessing (DataForge)

Supported datasets:

- NSL‑KDD: see [UNB NSL‑KDD](https://www.unb.ca/cic/datasets/nsl.html)
- CIC‑IDS2017: see [CIC‑IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

Place raw datasets under a data root (or point `--data-root` to where they reside). The preprocessing CLI will handle loading, cleaning, encoding, scaling, and minority‑class identification.

Run preprocessing:
```bash
# Choose one: nsl_kdd | cic_ids2017 | unified (merges both)
python scripts/preprocess_data.py --data-root /path/to/datasets --dataset nsl_kdd
python scripts/preprocess_data.py --data-root /path/to/datasets --dataset cic_ids2017
python scripts/preprocess_data.py --data-root /path/to/datasets --dataset unified
```
Example output includes the preprocessed array shape and detected minority classes (<1%).

## 3) Training (ForgeTrain)

The main training entrypoint is `scripts/train_gan.py`, which forwards to `train/train_stealthgan.py`.

CLI arguments:

- `--data-root`: path to datasets (default: current working directory)
- `--dataset`: `nsl_kdd` | `cic_ids2017` | `unified` (default: `unified`)
- `--epochs`: number of epochs (default: 10)
- `--checkpoint-interval`: save a checkpoint every N epochs (default: 10)
- `--resume`: path to a prior checkpoint to resume from
- `--cpu`: force CPU even if CUDA is available

Run training (examples):
```bash
# Train on NSL‑KDD
python scripts/train_gan.py --data-root /path/to/datasets --dataset nsl_kdd --epochs 50

# Train on CIC‑IDS2017
python scripts/train_gan.py --data-root /path/to/datasets --dataset cic_ids2017 --epochs 50

# Train on unified (NSL‑KDD + CIC‑IDS2017)
python scripts/train_gan.py --data-root /path/to/datasets --dataset unified --epochs 100

# Resume from a checkpoint
python scripts/train_gan.py --data-root /path/to/datasets --dataset unified \
  --epochs 100 --resume checkpoint_epoch_50.pth
```

Outputs saved to the working directory:

- `training_stats.csv`: per‑epoch loss and sample stats
- `checkpoint_epoch_*.pth`: periodic checkpoints with model + optimizer + config
- `generator_best.pth`: best generator snapshot (by generator loss)
- `generator.pth`, `discriminator.pth`: final weights

Training defaults (see `utils/config.py`):

- `latent_dim=100`, `batch_size=256`, `critic_updates=3`, `gp_lambda=20.0`
- `lr_g=5e-5`, `lr_d=5e-5`, ReduceLROnPlateau schedulers
- Early stopping with `early_stopping_patience=170`
- Label smoothing and optional noisy labels
- Feature matching regularization

## 4) Evaluation and IDS Uplift (AugmentEval)

Use `scripts/eval_gan.py` to generate synthetic samples, compute KL/JS divergence, create t‑SNE plots, and evaluate IDS uplift via Random Forest classifiers.

CLI arguments:

- `--data-root`: path to datasets (default: cwd)
- `--dataset`: `nsl_kdd` | `cic_ids2017` | `unified` (default: `unified`)
- `--generator-path`: path to generator weights (`generator_best.pth` or full checkpoint)
- `--target-minority`: oversample rare classes when generating synthetic data
- `--n-per-class`: synthetic samples per rare class (default: 2000)
- `--tsne-outdir`: directory to save plots (default: `eval_outputs/plots/tsne`)

Quick start:
```bash
# Evaluate with default generator auto‑selection
python scripts/eval_gan.py --data-root /path/to/datasets --dataset nsl_kdd

# Evaluate with explicit generator
python scripts/eval_gan.py --data-root /path/to/datasets --dataset unified \
  --generator-path Merged\ Training\ Results/generator_best.pth --target-minority --n-per-class 3000
```

Artifacts:

- t‑SNE plots: real vs synthetic overlap, per‑class overlays (saved under `--tsne-outdir`)
- Divergence metrics: KL/JS printed to console
- IDS uplift: baseline vs augmented Random Forest performance (F1/AUC/accuracy)
- Saved classifiers: `rf_baseline.pkl`, `rf_augmented.pkl`

## 5) Experiment Tracking (TrackForge)

`utils/logging.ExperimentLogger` integrates with [MLflow](https://mlflow.org/) if installed:

- Parameters and metrics are logged per run
- Artifacts (you can manually log files) can be tracked
- Local runs are stored in `mlruns/`

Launch MLflow UI:
```bash
mlflow ui --backend-store-uri mlruns
# Then open http://127.0.0.1:5000 in your browser
```

If `mlflow` is not installed, logging gracefully becomes a no‑op.

## 6) How the Model Works (StealthGAN)

- Class‑conditioned Generator: `z ∈ R^{100}` + class embedding → dense + self‑attention + 1D upsampling → synthetic feature vector
- Critic/Discriminator: 1D convs + self‑attention + WGAN critic score and auxiliary class prediction
- Losses: WGAN‑GP for stability, auxiliary CE loss for class guidance, optional feature matching

See architecture figures in `!Figs/` (e.g., `generator_architecture.png`, `discriminator_architecture.png`, `loss_functions.png`).

## 7) Reproducibility & Checkpoints

- Checkpoints contain model/optimizer/scheduler states and the config snapshot
- Use `--resume` to continue training from a checkpoint
- Generators saved as `generator_best.pth` (weights only) and `generator.pth` (final). The evaluator accepts both a weights file or a full checkpoint (it detects and loads accordingly).

## 8) Troubleshooting

- Install errors: ensure you’re using the correct PyTorch wheel for your CUDA toolkit. See the official [PyTorch install guide](https://pytorch.org/get-started/locally/).
- Out‑of‑memory on GPU: reduce `batch_size` in `utils/config.py` or run with `--cpu`.
- Divergence or unstable losses: increase `gp_lambda`, reduce `lr_g/lr_d`, or decrease `critic_updates`.
- Empty/NaN plots: ensure `matplotlib` backend is working and there’s enough data after preprocessing.
- `mlflow` not logging: verify `pip show mlflow` and that `mlflow ui` runs without errors.

## 9) At‑a‑Glance Commands

```bash
# 0) Environment
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 1) Preprocess
python scripts/preprocess_data.py --data-root /data/ids --dataset unified

# 2) Train
python scripts/train_gan.py --data-root /data/ids --dataset unified --epochs 100

# 3) Evaluate (auto‑select generator or pass an explicit path)
python scripts/eval_gan.py --data-root /data/ids --dataset unified --target-minority --n-per-class 3000
```

## 10) Citation

If you use this repository, please cite the core methods: [WGAN‑GP](https://arxiv.org/abs/1704.00028) and [AC‑GAN](https://arxiv.org/abs/1610.09585). Project: StealthGAN‑IDS (2025).