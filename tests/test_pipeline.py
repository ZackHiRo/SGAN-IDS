import pandas as pd

from sgan_ids.config import load_config
from sgan_ids.experiment import run


def test_toy_experiment_end_to_end(tmp_path):
    cfg = load_config(
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
    run(cfg)
    results = cfg.run_dir / "results"
    det = pd.read_csv(results / "detection.csv")
    assert set(det["method"]) == set(cfg.augment.methods)
    assert det["macro_f1"].between(0, 1).all()
    assert set(pd.read_csv(results / "privacy.csv")["method"]) == {"ros", "smote", "gan"}
    report = (cfg.run_dir / "report.md").read_text()
    assert "Ours (cWGAN-GP)" in report and (cfg.run_dir / "figures" / "per_class_f1.png").exists()
