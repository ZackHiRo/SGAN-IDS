import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from utils.metrics import kl_divergence, js_divergence
from data import DataForge
import torch
from models.generator import Generator
import joblib


def safe_normalize(x):
    x = np.clip(x, 0, None)
    s = x.sum()
    return x / s if s > 0 else np.ones_like(x) / len(x)


def evaluate(args):
    # Load and preprocess real data
    if args.dataset == "unified":
        datasets = ["nsl_kdd", "cic_ids2017"]
    else:
        datasets = [args.dataset]
    forge = DataForge(args.data_root).load(datasets)
    (X, y), _ = forge.preprocess()
    y = y.astype(int)
    if hasattr(X, "toarray"):
        X = X.toarray()
    n_samples, n_features = X.shape
    n_classes = int(y.max()) + 1
    latent_dim = 100  # Should match training config

    # --- Print real data stats ---
    print(f"[Real data] Mean: {X.mean():.4f}, Std: {X.std():.4f}")

    # --- Load trained generator (strict, no fallback) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.generator_path, map_location=device)
    state_dict = ckpt["generator_state_dict"] if isinstance(ckpt, dict) and "generator_state_dict" in ckpt else ckpt
    # Infer class_embed_dim from checkpoint if present
    class_embed_dim = None
    if isinstance(state_dict, dict) and "embed.weight" in state_dict:
        class_embed_dim = state_dict["embed.weight"].shape[1]
    G = Generator(latent_dim, n_features, n_classes, class_embed_dim=class_embed_dim).to(device)
    # Strict load
    G.load_state_dict(state_dict)
    G.eval()

    # --- Generate synthetic data (optionally targeted for rare classes) ---
    synth_batch = 256
    synth = []
    y_synth = []
    if args.target_minority:
        # Oversample minority classes
        class_counts = np.bincount(y)
        total = sum(class_counts)
        rare_classes = np.where(class_counts / total < 0.01)[0]
        n_per_class = args.n_per_class
        for cls in rare_classes:
            n_gen = n_per_class
            while n_gen > 0:
                bsz = min(synth_batch, n_gen)
                z = torch.randn(bsz, latent_dim, device=device)
                labels = torch.full((bsz,), cls, dtype=torch.long, device=device)
                with torch.no_grad():
                    fake = G(z, labels).cpu().numpy()
                synth.append(fake)
                y_synth.append(labels.cpu().numpy())
                n_gen -= bsz
        # Optionally, add some majority class samples for balance
        n_major = min(10000, n_samples)
        n_gen = n_major
        while n_gen > 0:
            bsz = min(synth_batch, n_gen)
            z = torch.randn(bsz, latent_dim, device=device)
            labels = torch.randint(0, n_classes, (bsz,), device=device)
            with torch.no_grad():
                fake = G(z, labels).cpu().numpy()
            synth.append(fake)
            y_synth.append(labels.cpu().numpy())
            n_gen -= bsz
    else:
        n_synth = min(10000, n_samples)
        while len(synth) < n_synth:
            bsz = min(synth_batch, n_synth - len(synth))
            z = torch.randn(bsz, latent_dim, device=device)
            labels = torch.randint(0, n_classes, (bsz,), device=device)
            with torch.no_grad():
                fake = G(z, labels).cpu().numpy()
            synth.append(fake)
            y_synth.append(labels.cpu().numpy())
    synth = np.vstack(synth)
    y_synth = np.hstack(y_synth)

    # --- Print synthetic data stats ---
    print(f"[Synthetic data] Mean: {synth.mean():.4f}, Std: {synth.std():.4f}")

    # KL/JS divergence (safe normalization)
    real_dist = safe_normalize(X.mean(0))
    synth_dist = safe_normalize(synth.mean(0))
    kl = kl_divergence(real_dist, synth_dist)
    js = js_divergence(real_dist, synth_dist)
    print(f"KL div: {kl:.4f} | JS div: {js:.4f}")

    # Prepare t-SNE output directory
    tsne_outdir = Path(args.tsne_outdir) if hasattr(args, "tsne_outdir") and args.tsne_outdir else Path.cwd()
    tsne_outdir.mkdir(parents=True, exist_ok=True)

    # t-SNE visualization (overall)
    tsne = TSNE(n_components=2, init="random", random_state=42)
    emb = tsne.fit_transform(np.vstack([X[:500], synth[:500]]))
    plt.figure(figsize=(7, 5))
    plt.scatter(emb[:500, 0], emb[:500, 1], c="#1f77b4", alpha=0.6, label="Real", marker='o')
    plt.scatter(emb[500:, 0], emb[500:, 1], c="#d62728", alpha=0.6, label="Synthetic", marker='x')
    plt.legend(); plt.title("t-SNE: Real vs Synthetic"); plt.tight_layout()
    out_path = tsne_outdir / "tsne_real_vs_synth.png"
    plt.savefig(out_path.as_posix())
    print(f"t-SNE plot saved as {out_path}")
    plt.close()

    # Combined all-classes t-SNE (class-specific colors; 'o' real, 'x' synthetic)
    sample_n = min(2000, min(len(X), len(synth)))
    tsne_all = TSNE(n_components=2, init="random", random_state=42)
    emb_all = tsne_all.fit_transform(np.vstack([X[:sample_n], synth[:sample_n]]))
    cmap = plt.cm.get_cmap("tab20", n_classes)
    plt.figure(figsize=(9, 7))
    # Real per-class
    for cls in range(n_classes):
        idx = np.where(y[:sample_n] == cls)[0]
        if idx.size == 0:
            continue
        plt.scatter(emb_all[idx, 0], emb_all[idx, 1], color=cmap(cls), alpha=0.55, marker='o', label=f"Real c{cls}")
    # Synthetic per-class
    offset = sample_n
    for cls in range(n_classes):
        idx = np.where(y_synth[:sample_n] == cls)[0]
        if idx.size == 0:
            continue
        idx = idx + offset
        plt.scatter(emb_all[idx, 0], emb_all[idx, 1], color=cmap(cls), alpha=0.55, marker='x', label=f"Synth c{cls}")
    plt.title("t-SNE: All Classes (o=Real, x=Synthetic)")
    # Reduce legend clutter by making it outside and compact
    plt.tight_layout()
    out_all = tsne_outdir / "tsne_all_classes.png"
    plt.savefig(out_all.as_posix())
    print(f"t-SNE (all classes) saved as {out_all}")
    plt.close()

    # --- Per-class t-SNE visualization ---
    class_names = [str(i) for i in range(n_classes)]
    for class_idx, class_name in enumerate(class_names):
        real_idx = np.where(y == class_idx)[0]
        synth_idx = np.where(y_synth == class_idx)[0]
        if len(real_idx) < 5 or len(synth_idx) < 5:
            continue  # skip classes with too few samples
        real_sample = X[real_idx[:200]]
        synth_sample = synth[synth_idx[:200]]
        X_cat = np.vstack([real_sample, synth_sample])
        y_cat = np.array([0]*len(real_sample) + [1]*len(synth_sample))
        tsne = TSNE(n_components=2, random_state=42)
        X_emb = tsne.fit_transform(X_cat)
        plt.figure(figsize=(6,4))
        plt.scatter(X_emb[y_cat==0,0], X_emb[y_cat==0,1], c='#1f77b4', label='Real', alpha=0.6, marker='o')
        plt.scatter(X_emb[y_cat==1,0], X_emb[y_cat==1,1], c='#d62728', label='Synthetic', alpha=0.6, marker='x')
        plt.title(f't-SNE: Real vs Synthetic (Class: {class_name})')
        plt.legend()
        plt.tight_layout()
        plt.savefig((tsne_outdir / f'tsne_real_vs_synth_class_{class_name}.png').as_posix())
        plt.close()

    # --- Baseline classifier: real data only ---
    print("\n[Baseline: Real data only]")
    clf_base = RandomForestClassifier(n_estimators=50)
    n = len(X)
    X_train, y_train = X[:n//2], y[:n//2]
    X_test, y_test = X[n//2:], y[n//2:]
    # --- Print class distributions for train/test sets ---
    print(f"Train set class distribution: {np.bincount(y_train)}")
    print(f"Test set class distribution: {np.bincount(y_test)}")
    clf_base.fit(X_train, y_train)
    y_pred_base = clf_base.predict(X_test)
    print(classification_report(y_test, y_pred_base))
    try:
        auc_base = roc_auc_score(y_test, clf_base.predict_proba(X_test), multi_class="ovr")
        print(f"AUC: {auc_base:.4f}")
    except Exception:
        pass
    # Print summary metrics for baseline
    macro_f1_base = f1_score(y_test, y_pred_base, average="macro")
    weighted_f1_base = f1_score(y_test, y_pred_base, average="weighted")
    acc_base = accuracy_score(y_test, y_pred_base)
    print(f"[Baseline] Macro F1: {macro_f1_base:.4f}, Weighted F1: {weighted_f1_base:.4f}, Accuracy: {acc_base:.4f}")
    # Save baseline classifier
    joblib.dump(clf_base, "rf_baseline.pkl")
    print("[eval] Baseline classifier saved as rf_baseline.pkl")

    # --- Augmented classifier: real + synthetic ---
    print("\n[Augmented: Real + Synthetic]")
    clf = RandomForestClassifier(n_estimators=50)
    X_train_aug = np.vstack([X[:n//2], synth[:n//2]])
    y_train_aug = np.hstack([y[:n//2], y_synth[:n//2]])
    X_test_aug = X[n//2:]
    y_test_aug = y[n//2:]
    # --- Print class distributions for augmented train/test sets ---
    print(f"Augmented train set class distribution: {np.bincount(y_train_aug)}")
    print(f"Augmented test set class distribution: {np.bincount(y_test_aug)}")
    clf.fit(X_train_aug, y_train_aug)
    y_pred = clf.predict(X_test_aug)
    print(classification_report(y_test_aug, y_pred))
    try:
        auc = roc_auc_score(y_test_aug, clf.predict_proba(X_test_aug), multi_class="ovr")
        print(f"AUC: {auc:.4f}")
    except Exception:
        pass
    # Print summary metrics for augmented
    macro_f1_aug = f1_score(y_test_aug, y_pred, average="macro")
    weighted_f1_aug = f1_score(y_test_aug, y_pred, average="weighted")
    acc_aug = accuracy_score(y_test_aug, y_pred)
    print(f"[Augmented] Macro F1: {macro_f1_aug:.4f}, Weighted F1: {weighted_f1_aug:.4f}, Accuracy: {acc_aug:.4f}")
    # Save augmented classifier
    joblib.dump(clf, "rf_augmented.pkl")
    print("[eval] Augmented classifier saved as rf_augmented.pkl")


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path.cwd())
    p.add_argument("--generator-path", type=str, default="generator.pth")
    p.add_argument("--target-minority", action="store_true", help="Generate more synthetic samples for rare classes")
    p.add_argument("--n-per-class", type=int, default=2000, help="Number of synthetic samples per rare class")
    p.add_argument("--dataset", type=str, default="unified", choices=["nsl_kdd", "cic_ids2017", "unified"], help="Which dataset to evaluate on")
    p.add_argument("--tsne-outdir", type=str, default="eval_outputs/plots/tsne", help="Directory to save t-SNE plots")
    args = p.parse_args()
    # Default generator selection if not provided explicitly
    if args.generator_path == "generator.pth" or not args.generator_path:
        proj = Path(__file__).resolve().parents[1]
        if args.dataset == "nsl_kdd":
            candidate = proj / "NSL_KDD_LAST" / "generator_best.pth"
            args.generator_path = candidate.as_posix()
        elif args.dataset == "unified":
            candidate = proj / "Merged Training Results" / "generator_best.pth"
            args.generator_path = candidate.as_posix()
    evaluate(args)


if __name__ == "__main__":
    cli() 