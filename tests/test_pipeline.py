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
    assert "Do synthetic rows look like their own class?" in report
    fid = pd.read_csv(results / "fidelity.csv")
    assert fid["label_consistency"].between(0, 1).all()
    assert (fid[fid.method == "ros"]["label_consistency"] > 0.9).all()  # copies of real rows
    assert (cfg.run_dir / "figures" / "per_class_f1.png").exists()


def test_rerun_reuses_generators_and_synthetic_data(tmp_path, caplog):
    cfg = _cfg(tmp_path)
    run(cfg)
    first = pd.read_csv(cfg.run_dir / "results" / "detection.csv")
    with caplog.at_level(logging.INFO, logger="sgan_ids"):
        run(cfg)
    assert "already trained with this config and data" in caplog.text
    assert caplog.text.count("reusing cached synthetic rows") == 3  # ros, smote, gan
    pd.testing.assert_frame_equal(first, pd.read_csv(cfg.run_dir / "results" / "detection.csv"))


def test_generator_is_retrained_when_training_data_changes(tmp_path, caplog):
    run(_cfg(tmp_path))
    changed = _cfg(tmp_path)
    changed.data.test_size = 0.3  # different split -> different training rows
    with caplog.at_level(logging.INFO, logger="sgan_ids"):
        run(changed)
    assert "training data changed since seed 0 was trained, retraining" in caplog.text
    assert "reusing cached synthetic rows" not in caplog.text
