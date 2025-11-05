from __future__ import annotations

from typing import Any, Dict

try:
    import mlflow
except ImportError:  # graceful fallback
    mlflow = None


class ExperimentLogger:
    """Minimal MLflow wrapper (no-op if MLflow missing)."""

    def __init__(self, exp_name: str = "StealthGAN-IDS"):
        self.enabled = mlflow is not None
        if self.enabled:
            mlflow.set_experiment(exp_name)
            self._run = mlflow.start_run(run_name=exp_name)
        else:
            self._run = None

    # -------------------------------------------------------
    def log_params(self, params: Dict[str, Any]):
        if self.enabled:
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None):
        if self.enabled:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str):
        if self.enabled:
            mlflow.log_artifact(path)

    def end(self):
        if self.enabled and self._run is not None:
            mlflow.end_run()

    # context-manager sugar
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end()
        return False 