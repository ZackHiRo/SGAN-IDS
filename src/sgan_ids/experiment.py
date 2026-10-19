"""End-to-end experiment: prepare -> train GAN per seed -> augment -> evaluate.

Layout of ``<output_dir>/<name>/``::

    config.yaml               resolved config of the run
    data/                     encoded splits + fitted transformer (shared by every step)
    gan/seed_<s>/             generator.pt (best validation checkpoint), history.csv
    results/detection.csv     one row per (seed, method, classifier), scored on the real test split
    results/fidelity.csv      one row per (seed, method, augmented class)
    results/privacy.csv       one row per (seed, method, augmented class)
    results/tstr.csv          rare classes replaced by synthetic rows only
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sgan_ids.augment import GENERATIVE, augment, augmentation_plan, resolve_classes
from sgan_ids.config import ExperimentConfig, save_config
from sgan_ids.data import PreparedData, prepare
from sgan_ids.evaluation.detection import bootstrap_ci, detection_metrics, fit_predict
from sgan_ids.evaluation.fidelity import class_fidelity
from sgan_ids.evaluation.privacy import dcr_ratio
from sgan_ids.training import load_generator, train_gan
from sgan_ids.utils import logger, resolve_device


def prepared_data(cfg: ExperimentConfig, refresh: bool = False) -> PreparedData:
    folder = cfg.run_dir / "data"
    if (folder / "splits.npz").exists() and not refresh:
        return PreparedData.load(folder)
    data = prepare(cfg.data, seed=cfg.seeds[0])
    data.save(folder)
    logger.info("Prepared %s: %s", cfg.data.dataset, json.dumps(data.report["rows"]))
    return data


def generator_path(cfg: ExperimentConfig, seed: int) -> Path:
    return cfg.run_dir / "gan" / f"seed_{seed}" / "generator.pt"


def train_all(cfg: ExperimentConfig, data: PreparedData, retrain: bool = False) -> None:
    for seed in cfg.seeds:
        path = generator_path(cfg, seed)
        if path.exists() and not retrain:
            saved = torch.load(path, map_location="cpu", weights_only=False)["config"]
            if saved == json.loads(json.dumps(dataclasses.asdict(cfg.gan))):
                logger.info(
                    "Generator for seed %d already trained with this config, reusing it", seed
                )
                continue
            logger.info("GAN config changed since seed %d was trained, retraining", seed)
        train_gan(data, cfg.gan, seed=seed, out_dir=path.parent)


def _append(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def evaluate_all(cfg: ExperimentConfig, data: PreparedData) -> None:
    out = cfg.run_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("detection.csv", "fidelity.csv", "privacy.csv", "tstr.csv"):
        (out / name).unlink(missing_ok=True)

    selected = resolve_classes(cfg.augment.classes, data.classes)
    plan = augmentation_plan(data.y_train, data.n_classes, cfg.augment.target_per_class, selected)
    rare = selected if selected is not None else sorted(plan)
    logger.info(
        "Augmenting %d classes to %d rows: %s",
        len(rare),
        cfg.augment.target_per_class,
        {data.classes[c]: n for c, n in plan.items()},
    )
    with open(out / "plan.json", "w") as f:
        json.dump({data.classes[c]: n for c, n in plan.items()}, f, indent=2)

    device = resolve_device(cfg.gan.device)
    for seed in cfg.seeds:
        generator = None
        if "gan" in cfg.augment.methods:
            generator = load_generator(generator_path(cfg, seed), device)

        for method in cfg.augment.methods:
            logger.info("seed %d | method %s", seed, method)
            aug = augment(
                method,
                data,
                plan,
                seed,
                generator,
                cfg.augment.ctgan_epochs,
                cfg.augment.ctgan_batch_size,
            )

            rows = []
            for clf in cfg.eval.classifiers:
                pred = fit_predict(clf, seed, aug.X, aug.y, data.X_test, aug.sample_weight)
                if pred is None:
                    logger.info("  %s does not support sample weights, skipped", clf)
                    continue
                metrics = detection_metrics(data.y_test, pred, data.n_classes, rare)
                lo, hi = bootstrap_ci(data.y_test, pred, data.n_classes, cfg.eval.bootstrap, seed)
                logger.info(
                    "  %-4s macro-F1 %.4f  rare-F1 %.4f",
                    clf,
                    metrics["macro_f1"],
                    metrics["rare_macro_f1"],
                )
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "classifier": clf,
                        "macro_f1_ci_low": lo,
                        "macro_f1_ci_high": hi,
                        **{
                            k: json.dumps(v) if isinstance(v, list) else v
                            for k, v in metrics.items()
                        },
                    }
                )
            _append(out / "detection.csv", rows)

            if method in GENERATIVE and aug.X_synth is not None:
                _synthetic_quality(cfg, data, aug, method, seed, rare, out)


def _synthetic_quality(cfg, data, aug, method, seed, rare, out) -> None:
    fidelity, privacy = [], []
    for c in np.unique(aug.y_synth):
        synth = aug.X_synth[aug.y_synth == c]
        test_c = data.X_test[data.y_test == c]
        train_c = data.X_train[data.y_train == c]
        val_c = data.X_val[data.y_val == c]
        base = {"seed": seed, "method": method, "class": data.classes[c]}
        if len(test_c) >= 10:
            fidelity.append(
                {
                    **base,
                    **class_fidelity(
                        test_c, synth, cfg.eval.knn_k, cfg.eval.fidelity_max_samples, seed
                    ),
                }
            )
        if len(val_c) >= 2:
            privacy.append(
                {**base, **dcr_ratio(train_c, val_c, synth, cfg.eval.fidelity_max_samples, seed)}
            )
    _append(out / "fidelity.csv", fidelity)
    _append(out / "privacy.csv", privacy)

    # Train-on-synthetic for the augmented classes: drop their real rows entirely.
    keep = ~np.isin(data.y_train, np.unique(aug.y_synth))
    X = np.vstack([data.X_train[keep], aug.X_synth])
    y = np.r_[data.y_train[keep], aug.y_synth]
    pred = fit_predict("rf", seed, X, y, data.X_test)
    m = detection_metrics(data.y_test, pred, data.n_classes, rare)
    _append(
        out / "tstr.csv",
        [
            {
                "seed": seed,
                "method": method,
                "macro_f1": m["macro_f1"],
                "rare_macro_f1": m["rare_macro_f1"],
                "rare_macro_recall": m["rare_macro_recall"],
            }
        ],
    )


def run(
    cfg: ExperimentConfig, steps: tuple[str, ...] = ("prepare", "train", "evaluate", "report")
) -> None:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, cfg.run_dir / "config.yaml")
    data = prepared_data(cfg, refresh="prepare" in steps)
    if "train" in steps and "gan" in cfg.augment.methods:
        train_all(cfg, data)
    if "evaluate" in steps:
        evaluate_all(cfg, data)
    if "report" in steps:
        from sgan_ids.report import build_report

        build_report(cfg.run_dir)
