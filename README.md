## StealthGAN‑IDS

StealthGAN‑IDS is an end‑to‑end pipeline to synthesize realistic, under‑represented network attacks for IDS augmentation. It implements:

- Data ingestion and preprocessing (DataForge)
- A class‑conditioned WGAN‑GP with auxiliary classification and self‑attention (StealthGAN)
- Training orchestration with checkpointing and experiment logging (ForgeTrain)
- Evaluation and augmentation tests with t‑SNE and IDS uplift (AugmentEval)

Key references: [WGAN‑GP](https://arxiv.org/abs/1704.00028), [AC‑GAN](https://arxiv.org/abs/1610.09585), [t‑SNE](https://jmlr.org/papers/v9/vandermaaten08a.html), [MLflow](https://mlflow.org/).

### Recent Improvements (v2.0)

This version includes significant architectural and methodological improvements:

**Architecture:**
- LayerNorm instead of BatchNorm (better with gradient penalty)
- Spectral normalization in discriminator for stability
- Residual blocks in generator for better gradient flow
- Improved minibatch discrimination (vectorized)

**Training:**
- Proper train/val/test splits with stratification
- Validation‑based early stopping (not training loss)
- EMA (Exponential Moving Average) for generator
- Mixed‑precision training (AMP) support
- Reproducibility via seeding

**Evaluation:**
- Multiple classifiers (Random Forest, XGBoost, LightGBM, MLP)
- Stratified k‑fold cross‑validation
- Statistical significance testing (Wilcoxon, bootstrap CI)
- Distribution metrics (MMD, precision/recall, coverage/density)
- Per‑class quality analysis

**Datasets:**
- CIC‑IDS2018 support (modern)
- UNSW‑NB15 support (modern)
- Proper data leakage prevention (scaler fitted on train only)

### Folder structure
```text
stealthgan_ids/
├── configs/               # YAML configuration files
├── data/                  # DataForge: loading, cleaning, encoding
├── models/                # Generator, Discriminator, Self‑Attention
├── train/                 # ForgeTrain: training loop & CLI
├── eval/                  # AugmentEval: metrics, plots, IDS tests
├── utils/                 # Metrics, config, MLflow logging
├── scripts/               # CLI entry points (preprocess/train/eval)
├── mlruns/                # Local MLflow runs (if mlflow installed)
└── eval_outputs/          # Generated plots & reports
```

## 0) Quick Start with Google Colab

For faster training on free GPU, use our [Colab Notebook](StealthGAN_IDS_Colab.ipynb):

1. Open `StealthGAN_IDS_Colab.ipynb` in Google Colab
2. Update the repository URL (or upload your code)
3. Select your dataset
4. Run all cells

The notebook handles:
- GPU setup and dependency installation
- Dataset download and preprocessing
- Training with automatic checkpointing
- Evaluation and visualization
- Easy result download

**Note:** Colab free tier has 12-hour session limits. Use checkpoints to resume training.

## 1) Installation

Requirements:

- Python 3.9+ (recommended)
- PyTorch 2.0+ (CPU or CUDA). Select the correct wheel from [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- Optional: [MLflow](https://mlflow.org/) for experiment tracking

Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt

# (Optional) For additional evaluation classifiers
pip install xgboost lightgbm

# (Optional) For YAML config support
pip install pyyaml

# (Optional) For experiment logging
pip install mlflow wandb
```

CUDA users: install the matching torch/torchvision with the appropriate CUDA version from the official instructions linked above.

## 2) Datasets and Preprocessing (DataForge)

Supported datasets:

| Dataset | Year | Description | Link |
|---------|------|-------------|------|
| NSL‑KDD | 2009 | Classic benchmark (legacy) | [UNB](https://www.unb.ca/cic/datasets/nsl.html) |
| CIC‑IDS2017 | 2017 | Modern attacks, labeled flows | [UNB](https://www.unb.ca/cic/datasets/ids-2017.html) |
| CIC‑IDS2018 | 2018 | Extended with more attack types | [UNB](https://www.unb.ca/cic/datasets/ids-2018.html) |
| UNSW‑NB15 | 2015 | Hybrid real/synthetic traffic | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |

Place raw datasets under a data root (or point `--data-root` to where they reside). The preprocessing handles:
- Loading and cleaning (NaN/Inf removal)
- Feature alignment across datasets
- Label encoding
- Train/val/test splitting with stratification
- Scaler fitting on training data only (no data leakage)

Run preprocessing:
```bash
# Choose one: nsl_kdd | cic_ids2017 | cic_ids2018 | unsw_nb15 | unified
python scripts/preprocess_data.py --data-root /path/to/datasets --dataset nsl_kdd
python scripts/preprocess_data.py --data-root /path/to/datasets --dataset cic_ids2018
python scripts/preprocess_data.py --data-root /path/to/datasets --dataset unified
```
Example output includes the preprocessed array shape, class distribution, and detected minority classes (<1%).

## 3) Training (ForgeTrain)

The main training entrypoint is `scripts/train_gan.py`, which forwards to `train/train_stealthgan.py`.

CLI arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | cwd | Path to datasets |
| `--dataset` | `unified` | `nsl_kdd`, `cic_ids2017`, `cic_ids2018`, `unsw_nb15`, or `unified` |
| `--epochs` | 100 | Number of training epochs |
| `--checkpoint-interval` | 10 | Save checkpoint every N epochs |
| `--resume` | None | Path to checkpoint to resume from |
| `--cpu` | False | Force CPU training |
| `--amp` | False | Enable mixed‑precision training (faster on modern GPUs) |
| `--seed` | 42 | Random seed for reproducibility |
| `--num-workers` | 4 | DataLoader workers |

Run training (examples):
```bash
# Train on modern dataset with mixed-precision
python scripts/train_gan.py --data-root /path/to/datasets --dataset cic_ids2018 --epochs 100 --amp

# Train on UNSW-NB15
python scripts/train_gan.py --data-root /path/to/datasets --dataset unsw_nb15 --epochs 100

# Train on unified (NSL‑KDD + CIC‑IDS2017) - legacy
python scripts/train_gan.py --data-root /path/to/datasets --dataset unified --epochs 100

# Resume from a checkpoint with specific seed
python scripts/train_gan.py --data-root /path/to/datasets --dataset cic_ids2018 \
  --epochs 200 --resume checkpoint_epoch_100.pth --seed 123
```

Outputs saved to the working directory:

- `training_stats.csv`: per‑epoch training and validation metrics
- `checkpoint_epoch_*.pth`: periodic checkpoints (model + optimizer + EMA + config)
- `generator_best.pth`: best generator (by validation loss)
- `generator_ema_best.pth`: EMA version of best generator (often better quality)
- `generator.pth`, `discriminator.pth`: final weights

Training defaults (see `utils/config.py` or `configs/default_config.yaml`):

- Architecture: `latent_dim=100`, residual blocks, LayerNorm, spectral normalization
- Optimization: `batch_size=256`, `critic_updates=5`, `gp_lambda=10.0`
- Learning rates: `lr_g=2e-4`, `lr_d=2e-4` with ReduceLROnPlateau (based on validation)
- Regularization: `dropout=0.1`, `label_smoothing=0.1`, `feature_matching_weight=1.0`
- EMA: `ema_decay=0.999` (use `generator_ema_best.pth` for generation)
- Early stopping: `patience=30` epochs based on **validation** loss (not training)

## 4) Evaluation and IDS Uplift (AugmentEval)

Use `scripts/eval_gan.py` for comprehensive evaluation including:
- Multiple classifiers (Random Forest, XGBoost, LightGBM, MLP)
- Stratified k‑fold cross‑validation
- Statistical significance testing
- Distribution quality metrics (MMD, precision/recall, coverage/density)
- Per‑class quality analysis
- t‑SNE visualization

CLI arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | cwd | Path to datasets |
| `--dataset` | `unified` | Dataset to evaluate on |
| `--generator-path` | **required** | Path to generator checkpoint |
| `--target-minority` | False | Oversample rare classes |
| `--n-per-class` | 2000 | Synthetic samples per class |
| `--minority-threshold` | 0.01 | Threshold for minority class (fraction) |
| `--cv-folds` | 5 | Number of cross‑validation folds |
| `--n-estimators` | 100 | Estimators for tree classifiers |
| `--output-dir` | `eval_outputs` | Directory for results |

Quick start:
```bash
# Full evaluation with EMA generator
python scripts/eval_gan.py --data-root /path/to/datasets --dataset cic_ids2018 \
  --generator-path generator_ema_best.pth --target-minority

# Evaluate on UNSW-NB15 with more samples
python scripts/eval_gan.py --data-root /path/to/datasets --dataset unsw_nb15 \
  --generator-path checkpoint_epoch_100.pth --n-per-class 5000 --cv-folds 10
```

Outputs (saved to `--output-dir`):

- `evaluation_results.json`: Complete metrics in JSON format
- `plots/tsne_real_vs_synth.png`: t‑SNE visualization
- `plots/tsne_by_class.png`: Per‑class t‑SNE comparison

Console output includes:
- Distribution metrics: KL/JS divergence, MMD, precision/recall, coverage/density
- Per‑class quality: MMD and precision/recall per attack type
- Classifier results: Baseline vs augmented with confidence intervals
- Statistical significance: Wilcoxon test p‑values and effect sizes

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

### Generator Architecture
```
z ∈ R^{100} + class_embedding → [Linear → LayerNorm → LeakyReLU] × 4
                              → ResidualBlock × 2
                              → Linear → Tanh → Self‑Attention
                              → synthetic feature vector
```

### Discriminator Architecture
```
input features → [Linear → LayerNorm → LeakyReLU] × 3 (with spectral norm)
              → Self‑Attention
              → Minibatch Discrimination
              → [Wasserstein score, class logits]
```

### Loss Functions
- **WGAN‑GP**: Wasserstein distance with gradient penalty for stability
- **Auxiliary CE**: Cross‑entropy loss for class conditioning
- **Feature Matching**: L2 distance between real/fake feature statistics

### Key Improvements Over Standard AC‑GAN
1. **LayerNorm** instead of BatchNorm (works with gradient penalty)
2. **Spectral Normalization** for Lipschitz constraint
3. **Residual Blocks** for better gradient flow in deeper networks
4. **EMA** for smoother, higher‑quality generator weights
5. **Projection Discriminator** for class conditioning

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
# 0) Environment setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install xgboost lightgbm pyyaml  # Optional: for full evaluation

# 1) Preprocess (view data statistics)
python scripts/preprocess_data.py --data-root /data/ids --dataset cic_ids2018

# 2) Train with mixed-precision and EMA
python scripts/train_gan.py \
  --data-root /data/ids \
  --dataset cic_ids2018 \
  --epochs 100 \
  --amp \
  --seed 42

# 3) Evaluate with cross-validation and statistical tests
python scripts/eval_gan.py \
  --data-root /data/ids \
  --dataset cic_ids2018 \
  --generator-path generator_ema_best.pth \
  --target-minority \
  --n-per-class 3000 \
  --cv-folds 5

# 4) View results
cat eval_outputs/evaluation_results.json
```

### Quick Dataset Download

For modern datasets, you can use:
```bash
# CIC-IDS2018 (requires Kaggle account)
kaggle datasets download -d solarmainframe/ids-intrusion-csv

# UNSW-NB15
# Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset
```

## 10) Citation

If you use this repository, please cite the core methods: [WGAN‑GP](https://arxiv.org/abs/1704.00028) and [AC‑GAN](https://arxiv.org/abs/1610.09585). Project: StealthGAN‑IDS (2025).