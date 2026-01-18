"""Comprehensive evaluation module for StealthGAN-IDS.

Features:
- Multiple classifiers (Random Forest, XGBoost, LightGBM, MLP)
- Stratified k-fold cross-validation
- Statistical significance testing (Wilcoxon, bootstrap CI)
- Distribution quality metrics (MMD, precision/recall, coverage/density)
- Per-class quality analysis
- t-SNE and UMAP visualization
"""

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch

from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from scipy import stats

from data import DataForge
from models.generator import Generator
from utils.metrics import (
    kl_divergence, js_divergence, mmd_rbf, mmd_linear,
    compute_precision_recall, compute_coverage_density,
    per_class_quality_metrics, wasserstein_per_feature,
    feature_correlation_distance,
)
from utils.config import EvalConfig

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    # Distribution metrics
    kl_divergence: float
    js_divergence: float
    mmd_rbf: float
    mmd_linear: float
    precision: float
    recall: float
    coverage: float
    density: float
    correlation_distance: float
    
    # Per-class metrics
    per_class_metrics: Dict[int, Dict[str, float]]
    
    # Classifier results
    baseline_results: Dict[str, Dict[str, float]]
    augmented_results: Dict[str, Dict[str, float]]
    
    # Statistical tests
    statistical_tests: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "distribution_metrics": {
                "kl_divergence": self.kl_divergence,
                "js_divergence": self.js_divergence,
                "mmd_rbf": self.mmd_rbf,
                "mmd_linear": self.mmd_linear,
                "precision": self.precision,
                "recall": self.recall,
                "coverage": self.coverage,
                "density": self.density,
                "correlation_distance": self.correlation_distance,
            },
            "per_class_metrics": {str(k): v for k, v in self.per_class_metrics.items()},
            "baseline_results": self.baseline_results,
            "augmented_results": self.augmented_results,
            "statistical_tests": self.statistical_tests,
        }
    
    def save(self, path: str | Path):
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=float)


def safe_normalize(x: np.ndarray) -> np.ndarray:
    """Safely normalize array to sum to 1."""
    x = np.clip(x, 0, None)
    s = x.sum()
    return x / s if s > 0 else np.ones_like(x) / len(x)


def get_classifiers(n_estimators: int = 100) -> Dict[str, Any]:
    """Get dictionary of classifiers to evaluate.
    
    Returns dict mapping name -> (classifier, needs_proba)
    """
    classifiers = {
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=500,
            early_stopping=True,
            random_state=42,
        ),
    }
    
    # Try to import XGBoost and LightGBM
    try:
        from xgboost import XGBClassifier
        classifiers["xgboost"] = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
    except ImportError:
        print("[eval] XGBoost not installed, skipping")
    
    try:
        from lightgbm import LGBMClassifier
        classifiers["lightgbm"] = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        print("[eval] LightGBM not installed, skipping")
    
    return classifiers


def cross_validate_classifier(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, List[float]]:
    """Perform stratified k-fold cross-validation.
    
    Returns dict with lists of per-fold scores.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    results = {
        "accuracy": [],
        "macro_f1": [],
        "weighted_f1": [],
        "macro_precision": [],
        "macro_recall": [],
        "auc": [],
    }
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Clone classifier for each fold
        from sklearn.base import clone
        clf_fold = clone(clf)
        clf_fold.fit(X_train, y_train)
        
        y_pred = clf_fold.predict(X_test)
        
        results["accuracy"].append(accuracy_score(y_test, y_pred))
        results["macro_f1"].append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        results["weighted_f1"].append(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        results["macro_precision"].append(precision_score(y_test, y_pred, average="macro", zero_division=0))
        results["macro_recall"].append(recall_score(y_test, y_pred, average="macro", zero_division=0))
        
        # AUC (multi-class)
        try:
            if hasattr(clf_fold, "predict_proba"):
                y_proba = clf_fold.predict_proba(X_test)
                auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
                results["auc"].append(auc)
        except Exception:
            pass
    
    return results


def compute_bootstrap_ci(
    scores: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval.
    
    Returns (mean, lower_ci, upper_ci)
    """
    scores = np.array(scores)
    n = len(scores)
    
    if n < 2:
        return float(scores.mean()), float(scores.mean()), float(scores.mean())
    
    # Bootstrap resampling
    bootstrap_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bootstrap_means.append(scores[idx].mean())
    
    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return float(scores.mean()), float(lower), float(upper)


def statistical_significance_test(
    baseline_scores: List[float],
    augmented_scores: List[float],
) -> Dict[str, float]:
    """Perform statistical significance tests.
    
    Uses Wilcoxon signed-rank test for paired samples.
    """
    baseline = np.array(baseline_scores)
    augmented = np.array(augmented_scores)
    
    # Wilcoxon signed-rank test
    try:
        statistic, p_value = stats.wilcoxon(baseline, augmented, alternative="less")
    except Exception:
        statistic, p_value = np.nan, np.nan
    
    # Effect size (paired Cohen's d)
    diff = augmented - baseline
    effect_size = diff.mean() / (diff.std() + 1e-10)
    
    return {
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p_value": float(p_value),
        "effect_size_cohens_d": float(effect_size),
        "mean_improvement": float(diff.mean()),
    }


def generate_synthetic_samples(
    generator: torch.nn.Module,
    n_classes: int,
    latent_dim: int,
    device: torch.device,
    n_per_class: int = 2000,
    target_classes: Optional[List[int]] = None,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic samples from the generator.
    
    Args:
        generator: Trained generator model
        n_classes: Number of classes
        latent_dim: Latent dimension
        device: Torch device
        n_per_class: Samples to generate per class
        target_classes: Specific classes to generate (None = all)
        batch_size: Generation batch size
        
    Returns:
        (X_synth, y_synth) arrays
    """
    generator.eval()
    
    if target_classes is None:
        target_classes = list(range(n_classes))
    
    synth_X = []
    synth_y = []
    
    with torch.no_grad():
        for cls in target_classes:
            n_remaining = n_per_class
            while n_remaining > 0:
                bsz = min(batch_size, n_remaining)
                z = torch.randn(bsz, latent_dim, device=device)
                labels = torch.full((bsz,), cls, dtype=torch.long, device=device)
                fake = generator(z, labels).cpu().numpy()
                synth_X.append(fake)
                synth_y.append(labels.cpu().numpy())
                n_remaining -= bsz
    
    return np.vstack(synth_X), np.hstack(synth_y)


def create_visualizations(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    y_real: np.ndarray,
    y_synth: np.ndarray,
    output_dir: Path,
    n_classes: int,
    n_samples: int = 2000,
    class_names: Optional[List[str]] = None,
):
    """Create t-SNE visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Subsample for efficiency
    n_real = min(n_samples, len(X_real))
    n_synth = min(n_samples, len(X_synth))
    
    idx_real = np.random.choice(len(X_real), n_real, replace=False)
    idx_synth = np.random.choice(len(X_synth), n_synth, replace=False)
    
    X_real_sub = X_real[idx_real]
    X_synth_sub = X_synth[idx_synth]
    y_real_sub = y_real[idx_real]
    y_synth_sub = y_synth[idx_synth]
    
    # Combined t-SNE
    X_combined = np.vstack([X_real_sub, X_synth_sub])
    
    print("[eval] Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    emb = tsne.fit_transform(X_combined)
    
    emb_real = emb[:n_real]
    emb_synth = emb[n_real:]
    
    # Plot 1: Real vs Synthetic (binary)
    plt.figure(figsize=(10, 8))
    plt.scatter(emb_real[:, 0], emb_real[:, 1], c="#1f77b4", alpha=0.5, 
                label="Real", marker='o', s=20)
    plt.scatter(emb_synth[:, 0], emb_synth[:, 1], c="#d62728", alpha=0.5, 
                label="Synthetic", marker='x', s=20)
    plt.legend(fontsize=12)
    plt.title("t-SNE: Real vs Synthetic", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "tsne_real_vs_synth.png", dpi=150)
    plt.close()
    
    # Plot 2: All classes colored
    cmap = plt.cm.get_cmap("tab20", n_classes)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Real samples
    for cls in range(n_classes):
        mask = y_real_sub == cls
        if mask.sum() > 0:
            label = class_names[cls] if class_names else f"Class {cls}"
            axes[0].scatter(emb_real[mask, 0], emb_real[mask, 1], 
                          c=[cmap(cls)], alpha=0.6, s=20, label=label)
    axes[0].set_title("Real Samples", fontsize=14)
    axes[0].legend(loc='upper right', fontsize=8, ncol=2)
    
    # Synthetic samples
    for cls in range(n_classes):
        mask = y_synth_sub == cls
        if mask.sum() > 0:
            label = class_names[cls] if class_names else f"Class {cls}"
            axes[1].scatter(emb_synth[mask, 0], emb_synth[mask, 1], 
                          c=[cmap(cls)], alpha=0.6, s=20, label=label)
    axes[1].set_title("Synthetic Samples", fontsize=14)
    axes[1].legend(loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / "tsne_by_class.png", dpi=150)
    plt.close()
    
    print(f"[eval] Visualizations saved to {output_dir}")


def evaluate(args):
    """Main evaluation function."""
    print("=" * 60)
    print("StealthGAN-IDS Evaluation")
    print("=" * 60)
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[eval] Using device: {device}")
    
    # --- Load Data ---
    print("\n[1/6] Loading and preprocessing data...")
    if args.dataset == "unified":
        datasets = ["nsl_kdd", "cic_ids2017"]
    else:
        datasets = [args.dataset]
    
    # Use max_samples for memory-constrained environments
    max_samples = getattr(args, 'max_samples', None)
    if max_samples:
        print(f"[eval] Limiting dataset to {max_samples} samples")
    
    forge = DataForge(args.data_root, max_samples=max_samples).load(datasets)
    data_split, _ = forge.preprocess(random_state=args.seed)
    
    # Use test set for evaluation
    X_real = data_split.X_test
    y_real = data_split.y_test
    n_features = data_split.n_features
    n_classes = data_split.n_classes
    
    print(f"[eval] Test set: {len(X_real)} samples, {n_features} features, {n_classes} classes")
    
    # Get class names if available
    class_names = None
    if data_split.label_encoder is not None:
        class_names = list(data_split.label_encoder.classes_)
    
    # --- Load Generator ---
    print("\n[2/6] Loading generator...")
    ckpt = torch.load(args.generator_path, map_location=device, weights_only=False)
    
    # Handle both checkpoint and state_dict formats
    if isinstance(ckpt, dict) and "generator_state_dict" in ckpt:
        state_dict = ckpt["generator_state_dict"]
        latent_dim = ckpt.get("config", {}).get("latent_dim", 100)
    else:
        state_dict = ckpt
        latent_dim = 100
    
    # Infer class embedding dim from checkpoint
    class_embed_dim = None
    if "embed.weight" in state_dict:
        class_embed_dim = state_dict["embed.weight"].shape[1]
    
    G = Generator(
        latent_dim=latent_dim,
        out_dim=n_features,
        n_classes=n_classes,
        class_embed_dim=class_embed_dim,
    ).to(device)
    G.load_state_dict(state_dict)
    G.eval()
    
    print(f"[eval] Generator loaded from {args.generator_path}")
    
    # --- Generate Synthetic Data ---
    print("\n[3/6] Generating synthetic samples...")
    
    # Identify minority classes
    class_counts = np.bincount(y_real, minlength=n_classes)
    total = class_counts.sum()
    minority_mask = (class_counts / total) < args.minority_threshold
    minority_classes = np.where(minority_mask)[0].tolist()
    
    print(f"[eval] Minority classes (< {args.minority_threshold*100:.1f}%): {minority_classes}")
    
    if args.target_minority and minority_classes:
        # Generate extra samples for minority classes
        X_synth, y_synth = generate_synthetic_samples(
            G, n_classes, latent_dim, device,
            n_per_class=args.n_per_class,
            target_classes=minority_classes,
        )
        # Also generate some for majority classes
        X_synth_maj, y_synth_maj = generate_synthetic_samples(
            G, n_classes, latent_dim, device,
            n_per_class=args.n_per_class // 2,
            target_classes=[c for c in range(n_classes) if c not in minority_classes],
        )
        X_synth = np.vstack([X_synth, X_synth_maj])
        y_synth = np.hstack([y_synth, y_synth_maj])
    else:
        X_synth, y_synth = generate_synthetic_samples(
            G, n_classes, latent_dim, device,
            n_per_class=args.n_per_class,
        )
    
    print(f"[eval] Generated {len(X_synth)} synthetic samples")
    print(f"[eval] Real stats - mean: {X_real.mean():.4f}, std: {X_real.std():.4f}")
    print(f"[eval] Synth stats - mean: {X_synth.mean():.4f}, std: {X_synth.std():.4f}")
    
    # --- Compute Distribution Metrics ---
    print("\n[4/6] Computing distribution quality metrics...")
    
    # Subsample for expensive metrics
    max_samples = 5000
    X_real_sub = X_real[:max_samples] if len(X_real) > max_samples else X_real
    X_synth_sub = X_synth[:max_samples] if len(X_synth) > max_samples else X_synth
    
    # Divergence metrics
    real_dist = safe_normalize(X_real_sub.mean(0))
    synth_dist = safe_normalize(X_synth_sub.mean(0))
    kl = kl_divergence(real_dist, synth_dist)
    js = js_divergence(real_dist, synth_dist)
    
    # MMD
    mmd_r = mmd_rbf(X_real_sub, X_synth_sub)
    mmd_l = mmd_linear(X_real_sub, X_synth_sub)
    
    # Precision/Recall
    precision, recall = compute_precision_recall(X_real_sub, X_synth_sub, k=5)
    
    # Coverage/Density
    coverage, density = compute_coverage_density(X_real_sub, X_synth_sub, k=5)
    
    # Correlation distance
    corr_dist = feature_correlation_distance(X_real_sub, X_synth_sub)
    
    print(f"[eval] KL divergence: {kl:.4f}")
    print(f"[eval] JS divergence: {js:.4f}")
    print(f"[eval] MMD (RBF): {mmd_r:.4f}")
    print(f"[eval] MMD (Linear): {mmd_l:.4f}")
    print(f"[eval] Precision: {precision:.4f}")
    print(f"[eval] Recall: {recall:.4f}")
    print(f"[eval] Coverage: {coverage:.4f}")
    print(f"[eval] Density: {density:.4f}")
    print(f"[eval] Correlation distance: {corr_dist:.4f}")
    
    # Per-class metrics
    print("\n[eval] Per-class quality metrics:")
    per_class = per_class_quality_metrics(
        X_real, X_synth, y_real, y_synth, n_classes
    )
    for cls, metrics in per_class.items():
        name = class_names[cls] if class_names else f"Class {cls}"
        if not np.isnan(metrics["mmd"]):
            print(f"  {name}: MMD={metrics['mmd']:.4f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
    
    # --- Classifier Evaluation ---
    print("\n[5/6] Evaluating classifiers with cross-validation...")
    
    classifiers = get_classifiers(n_estimators=args.n_estimators)
    
    # Prepare data
    X_train_real = data_split.X_train
    y_train_real = data_split.y_train
    
    # Augmented training set
    X_train_aug = np.vstack([X_train_real, X_synth])
    y_train_aug = np.hstack([y_train_real, y_synth])
    
    baseline_results = {}
    augmented_results = {}
    statistical_tests = {}
    
    for name, clf in classifiers.items():
        print(f"\n  Evaluating {name}...")
        
        # Baseline (real data only)
        baseline_cv = cross_validate_classifier(
            clf, X_train_real, y_train_real, 
            n_folds=args.cv_folds, random_state=args.seed
        )
        
        # Augmented (real + synthetic)
        augmented_cv = cross_validate_classifier(
            clf, X_train_aug, y_train_aug,
            n_folds=args.cv_folds, random_state=args.seed
        )
        
        # Compute summary statistics
        baseline_summary = {}
        augmented_summary = {}
        
        for metric in ["accuracy", "macro_f1", "weighted_f1"]:
            base_mean, base_lo, base_hi = compute_bootstrap_ci(baseline_cv[metric])
            aug_mean, aug_lo, aug_hi = compute_bootstrap_ci(augmented_cv[metric])
            
            baseline_summary[metric] = {
                "mean": base_mean,
                "ci_lower": base_lo,
                "ci_upper": base_hi,
            }
            augmented_summary[metric] = {
                "mean": aug_mean,
                "ci_lower": aug_lo,
                "ci_upper": aug_hi,
            }
        
        baseline_results[name] = baseline_summary
        augmented_results[name] = augmented_summary
        
        # Statistical significance test
        stat_test = statistical_significance_test(
            baseline_cv["macro_f1"],
            augmented_cv["macro_f1"]
        )
        statistical_tests[name] = stat_test
        
        # Print results
        base_f1 = baseline_summary["macro_f1"]["mean"]
        aug_f1 = augmented_summary["macro_f1"]["mean"]
        improvement = aug_f1 - base_f1
        p_val = stat_test["wilcoxon_p_value"]
        
        sig_marker = "*" if p_val < 0.05 else ""
        print(f"    Baseline Macro-F1: {base_f1:.4f}")
        print(f"    Augmented Macro-F1: {aug_f1:.4f} ({improvement:+.4f}){sig_marker}")
        if p_val < 0.05:
            print(f"    Statistically significant (p={p_val:.4f})")
    
    # --- Visualizations ---
    print("\n[6/6] Creating visualizations...")
    output_dir = Path(args.output_dir)
    create_visualizations(
        X_real, X_synth, y_real, y_synth,
        output_dir / "plots",
        n_classes,
        n_samples=args.tsne_samples,
        class_names=class_names,
    )
    
    # --- Save Results ---
    results = EvaluationResults(
        kl_divergence=kl,
        js_divergence=js,
        mmd_rbf=mmd_r,
        mmd_linear=mmd_l,
        precision=precision,
        recall=recall,
        coverage=coverage,
        density=density,
        correlation_distance=corr_dist,
        per_class_metrics=per_class,
        baseline_results=baseline_results,
        augmented_results=augmented_results,
        statistical_tests=statistical_tests,
    )
    
    results_path = output_dir / "evaluation_results.json"
    results.save(results_path)
    print(f"\n[eval] Results saved to {results_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nDistribution Quality:")
    print(f"  MMD (RBF): {mmd_r:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"\nClassifier Improvements (Macro-F1):")
    for name in classifiers.keys():
        base = baseline_results[name]["macro_f1"]["mean"]
        aug = augmented_results[name]["macro_f1"]["mean"]
        p_val = statistical_tests[name]["wilcoxon_p_value"]
        sig = "*" if p_val < 0.05 else ""
        print(f"  {name}: {base:.4f} -> {aug:.4f} ({aug-base:+.4f}){sig}")
    
    return results


def cli():
    p = argparse.ArgumentParser(description="Evaluate StealthGAN-IDS")
    
    # Data
    p.add_argument("--data-root", type=Path, default=Path.cwd())
    p.add_argument("--dataset", type=str, default="unified",
                   choices=["nsl_kdd", "cic_ids2017", "cic_ids2018", "unsw_nb15", "unified"])
    
    # Generator
    p.add_argument("--generator-path", type=str, required=True,
                   help="Path to trained generator checkpoint")
    
    # Generation
    p.add_argument("--target-minority", action="store_true",
                   help="Generate more samples for minority classes")
    p.add_argument("--n-per-class", type=int, default=2000,
                   help="Synthetic samples per class")
    p.add_argument("--minority-threshold", type=float, default=0.01,
                   help="Threshold for minority class (fraction)")
    
    # Evaluation
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Number of cross-validation folds")
    p.add_argument("--n-estimators", type=int, default=100,
                   help="Number of estimators for tree-based classifiers")
    
    # Output
    p.add_argument("--output-dir", type=str, default="eval_outputs",
                   help="Directory for output files")
    p.add_argument("--tsne-samples", type=int, default=2000,
                   help="Number of samples for t-SNE visualization")
    
    # Memory
    p.add_argument("--max-samples", type=int, default=None,
                   help="Max samples to load (for memory-constrained environments)")
    
    # Other
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    
    args = p.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluate(args)


if __name__ == "__main__":
    cli() 