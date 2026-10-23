"""Turn the result CSVs of a run into ``report.md`` and figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METHOD_ORDER = ["none", "class_weight", "ros", "smote", "ctgan", "gan"]
METHOD_LABEL = {
    "none": "No augmentation",
    "class_weight": "Class weights",
    "ros": "Random oversampling",
    "smote": "SMOTE",
    "ctgan": "CTGAN",
    "gan": "Ours (cWGAN-GP)",
}


def _ordered(methods) -> list[str]:
    return [m for m in METHOD_ORDER if m in set(methods)]


def _mean_std(frame: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    g = frame.groupby(by)[cols]
    mean, std = g.mean(), g.std().fillna(0.0)
    out = mean.copy().astype(object)
    for c in cols:
        out[c] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[c], std[c])]
    return out


def _md(frame: pd.DataFrame) -> str:
    frame = frame.reset_index()
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in frame.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def build_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    res = run_dir / "results"
    figs = run_dir / "figures"
    figs.mkdir(exist_ok=True)
    data_report = json.loads((run_dir / "data" / "data_report.json").read_text())
    classes = data_report["classes"]
    plan = json.loads((res / "plan.json").read_text())
    det = pd.read_csv(res / "detection.csv")
    methods = _ordered(det["method"])

    lines = [f"# Results: {run_dir.name}", ""]
    lines += [
        f"Dataset `{data_report['dataset']}`, split: {data_report['split']}. "
        f"Rows: {data_report['rows']}. Encoded features: {data_report['n_features_encoded']} "
        f"(GAN generation space: {data_report.get('n_features_generation', 'same')}).",
        "",
        f"Augmented classes (rows added): {plan}",
        "",
        f"Seeds: {sorted(det['seed'].unique().tolist())}. Values are mean ± std over seeds; "
        "all scores are on the real, untouched test split.",
        "",
    ]

    # ---- 1. detection -----------------------------------------------------
    cols = ["macro_f1", "rare_macro_f1", "rare_macro_recall", "balanced_accuracy"]
    table = _mean_std(det, ["classifier", "method"], cols)
    table = table.reindex(
        [(c, m) for c in det["classifier"].unique() for m in methods if (c, m) in table.index]
    )
    lines += ["## Detection (train on real + synthetic, test on real)", "", _md(table), ""]

    # Paired difference to the no-augmentation baseline (same seed, same classifier).
    if "none" in methods:
        base = det[det.method == "none"].set_index(["seed", "classifier"])["rare_macro_f1"]
        diffs = []
        for m in methods:
            if m == "none":
                continue
            cur = det[det.method == m].set_index(["seed", "classifier"])["rare_macro_f1"]
            d = (cur - base).dropna()
            diffs.append(
                {
                    "method": m,
                    "mean Δ rare-F1": f"{d.mean():+.3f}",
                    "seeds×classifiers improved": f"{int((d > 0).sum())}/{len(d)}",
                }
            )
        lines += [
            "### Rare-class F1 change vs. no augmentation (paired by seed and classifier)",
            "",
            _md(pd.DataFrame(diffs).set_index("method")),
            "",
        ]

    # Per-class F1 of the augmented classes, for each classifier.
    per_class = det.assign(f1=det["per_class_f1"].map(json.loads))
    rare_idx = [classes.index(c) for c in plan]
    for clf in per_class["classifier"].unique():
        sub = per_class[per_class.classifier == clf]
        rows = {}
        for m in methods:
            arr = np.array(sub[sub.method == m]["f1"].tolist())
            if len(arr):
                rows[METHOD_LABEL[m]] = {classes[i]: f"{arr[:, i].mean():.3f}" for i in rare_idx}
        lines += [
            f"### Per-class F1 on augmented classes ({clf})",
            "",
            _md(pd.DataFrame(rows).T.rename_axis("method")),
            "",
        ]
    _plot_per_class(per_class, methods, classes, rare_idx, figs / "per_class_f1.png")
    lines += ["![Per-class F1](figures/per_class_f1.png)", ""]

    if (res / "overlap.csv").exists():
        ov = pd.read_csv(res / "overlap.csv").set_index("class")
        pct = [
            "test_seen_with_other_label",
            "test_seen_with_same_label_only",
            "test_unseen",
            "train_shared_with_other_label",
        ]
        ov[pct] = ov[pct].map(lambda v: f"{100 * v:.1f} %")
        lines += [
            "### How separable are the rare classes?",
            "",
            "Share of each rare class's test rows whose exact feature vector also occurs in the "
            "training data under another label. No classifier can be sure of those rows, "
            "whatever augmentation is used.",
            "",
            _md(ov),
            "",
        ]

    # ---- 2. synthetic data quality ---------------------------------------
    if (res / "fidelity.csv").exists():
        fid = pd.read_csv(res / "fidelity.csv")
        priv = pd.read_csv(res / "privacy.csv")
        fcols = [
            "c2st_auc",
            "mmd",
            "precision",
            "recall",
            "coverage",
            "wasserstein",
            "correlation_gap",
        ]
        merged = fid.merge(priv, on=["seed", "method", "class"], how="left")
        agg = (
            merged.groupby(["method", "seed"])[fcols + ["dcr_ratio", "exact_copy_rate"]]
            .mean()
            .reset_index()
        )
        table = _mean_std(agg, ["method"], fcols + ["dcr_ratio", "exact_copy_rate"])
        table = table.reindex([m for m in METHOD_ORDER if m in table.index])
        lines += [
            "## Synthetic data quality (augmented classes, vs. real test rows)",
            "",
            "C2ST AUC: 0.5 = indistinguishable from real. DCR ratio: ~1 = as novel as real "
            "held-out traffic, 0 = copies of training rows.",
            "",
            _md(table),
            "",
        ]
        _plot_fidelity_privacy(agg, figs / "fidelity_vs_privacy.png")
        lines += ["![Fidelity vs privacy](figures/fidelity_vs_privacy.png)", ""]

        if "label_consistency" in fid.columns:
            lc = fid.pivot_table(index="method", columns="class", values="label_consistency")
            lc = lc.reindex([m for m in METHOD_ORDER if m in lc.index]).map(lambda v: f"{v:.2f}")
            ref = fid.groupby("class")["reference_recall"].mean()
            lc.loc["*real test rows*"] = [f"{ref[c]:.2f}" for c in lc.columns]
            lines += [
                "### Do synthetic rows look like their own class?",
                "",
                "Share of synthetic rows that a random forest trained on real data assigns to "
                "the intended class (mean over seeds). Rows assigned to another class teach "
                "the detector false positives. The last row is the same classifier's recall "
                "on real test rows of each class, for reference.",
                "",
                _md(lc.rename_axis("method")),
                "",
            ]

    gen_path = res / "generation.csv"
    if gen_path.exists() and gen_path.stat().st_size > 0:
        gen = pd.read_csv(gen_path)
        table = gen.groupby(["method", "class"])[
            ["requested", "generated", "condition_hit_rate"]
        ].mean()
        lines += [
            "### Conditional sampling",
            "",
            "Share of CTGAN rows whose generated label matched the requested class; "
            "mismatches are rejected and drawn again.",
            "",
            _md(table.round(4)),
            "",
        ]

    if (res / "tstr.csv").exists():
        tstr = pd.read_csv(res / "tstr.csv")
        table = _mean_std(tstr, ["method"], ["rare_macro_f1", "rare_macro_recall", "macro_f1"])
        table = table.reindex([m for m in METHOD_ORDER if m in table.index])
        lines += [
            "## Train on synthetic only (real rows of augmented classes removed, RF)",
            "",
            _md(table),
            "",
        ]

    # ---- 3. training curves ----------------------------------------------
    histories = sorted((run_dir / "gan").glob("seed_*/history.csv"))
    if histories:
        _plot_training(histories, figs / "training.png")
        lines += ["## GAN training", "", "![Training curves](figures/training.png)", ""]

    path = run_dir / "report.md"
    path.write_text("\n".join(lines))
    return path


def _plot_per_class(per_class, methods, classes, rare_idx, path) -> None:
    clf = "rf" if "rf" in set(per_class.classifier) else per_class.classifier.iloc[0]
    sub = per_class[per_class.classifier == clf]
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(rare_idx)), 4))
    x = np.arange(len(rare_idx))
    for k, m in enumerate(methods):
        arr = np.array(sub[sub.method == m]["f1"].tolist())[:, rare_idx]
        ax.bar(x + k * width, arr.mean(0), width, yerr=arr.std(0), label=METHOD_LABEL[m], capsize=2)
    ax.set_xticks(x + width * (len(methods) - 1) / 2, [classes[i] for i in rare_idx])
    ax.set_ylabel(f"F1 on real test ({clf})")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_fidelity_privacy(agg, path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    for m in _ordered(agg["method"]):
        s = agg[agg.method == m]
        ax.scatter(s["dcr_ratio"], s["c2st_auc"], label=METHOD_LABEL[m], s=40)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.axvline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("DCR ratio (→1 = novel rows)")
    ax.set_ylabel("C2ST AUC (→0.5 = realistic)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_training(histories, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for h in histories:
        df = pd.read_csv(h)
        label = h.parent.name
        axes[0].plot(df["step"], df["wasserstein_estimate"], label=label)
        axes[1].plot(df["step"], df["val_mmd"], label=label)
    axes[0].set_title("Critic Wasserstein estimate")
    axes[1].set_title("Validation MMD (class-averaged, selection metric)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("generator step")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
