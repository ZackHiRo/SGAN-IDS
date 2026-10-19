"""Optuna search over GAN hyperparameters.

The objective is the same validation MMD used for checkpoint selection, so
tuning only ever sees the train and validation splits. The test split stays
untouched until the final evaluation.
"""

from __future__ import annotations

import dataclasses
import json

from sgan_ids.config import ExperimentConfig
from sgan_ids.experiment import prepared_data
from sgan_ids.training import train_gan
from sgan_ids.utils import logger


def tune(cfg: ExperimentConfig, n_trials: int = 30) -> dict:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Tuning needs `pip install optuna` (or `pip install .[tune]`)") from exc

    data = prepared_data(cfg)
    seed = cfg.seeds[0]

    def objective(trial: optuna.Trial) -> float:
        width = trial.suggest_categorical("width", [128, 256, 512])
        gan = dataclasses.replace(
            cfg.gan,
            lr_g=trial.suggest_float("lr_g", 2e-5, 5e-4, log=True),
            lr_d=trial.suggest_float("lr_d", 2e-5, 5e-4, log=True),
            n_critic=trial.suggest_int("n_critic", 1, 5),
            gp_lambda=trial.suggest_float("gp_lambda", 1.0, 20.0, log=True),
            latent_dim=trial.suggest_categorical("latent_dim", [64, 128]),
            class_sampling_alpha=trial.suggest_float("class_sampling_alpha", 0.0, 1.0),
            generator_hidden=[width] * 3,
            critic_hidden=[width] * 2,
        )
        return train_gan(data, gan, seed=seed).best_score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    result = {"best_val_mmd": study.best_value, "params": study.best_params, "n_trials": n_trials}
    path = cfg.run_dir / "best_hyperparams.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    logger.info("Best trial: %s -> %s", json.dumps(study.best_params), path)
    return result
