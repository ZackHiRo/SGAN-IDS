import logging

import pandas as pd

from sgan_ids.config import load_config
from sgan_ids.experiment import run


def _cfg(tmp_path):
    return load_config(
        "configs/toy.yaml",
        [
            f"output_dir={tmp_path}",
            "seeds=[0]",
            "gan.n_steps=40",
            "gan.eval_every=20",
            "eval.classifiers=[rf]",
            "eval.bootstrap=10",
        ],
    )


def test_toy_experiment_end_to_end(tmp_path):
    cfg = _cfg(tmp_path)
    run(cfg)
    results = cfg.run_dir / "results"
    det = pd.read_csv(results / "detection.csv")
    assert set(det["method"]) == set(cfg.augment.methods)
    assert det["macro_f1"].between(0, 1).all()
    assert set(pd.read_csv(results / "privacy.csv")["method"]) == {"ros", "smote", "gan"}
    assert set(pd.read_csv(results / "overlap.csv")["class"]) == set(cfg.augment.classes)
    report = (cfg.run_dir / "report.md").read_text()
    assert "Ours (cWGAN-GP)" in report and "How separable" in report
    assert (cfg.run_dir / "figures" / "per_class_f1.png").exists()


def test_rerun_reuses_generators_and_synthetic_data(tmp_path, caplog):
    cfg = _cfg(tmp_path)
    run(cfg)
    first = pd.read_csv(cfg.run_dir / "results" / "detection.csv")
    with caplog.at_level(logging.INFO, logger="sgan_ids"):
        run(cfg)
    assert "already trained with this config" in caplog.text
    assert caplog.text.count("reusing cached synthetic rows") == 3  # ros, smote, gan
    pd.testing.assert_frame_equal(first, pd.read_csv(cfg.run_dir / "results" / "detection.csv"))
