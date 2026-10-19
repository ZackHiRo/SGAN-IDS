"""Command line entry point: ``sgan-ids <step> --config configs/<file>.yaml``."""

from __future__ import annotations

import argparse

from sgan_ids.config import load_config
from sgan_ids.utils import setup_logging

STEPS = ("prepare", "train", "evaluate", "report")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sgan-ids", description=__doc__)
    parser.add_argument(
        "step",
        choices=(*STEPS, "run", "tune"),
        help="'run' executes prepare, train, evaluate and report in order",
    )
    parser.add_argument("--config", "-c", required=True, help="YAML experiment config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set gan.n_steps=2000 (repeatable)",
    )
    parser.add_argument(
        "--trials", type=int, default=30, help="number of Optuna trials (tune only)"
    )
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config, args.overrides)

    if args.step == "tune":
        from sgan_ids.tuning import tune

        tune(cfg, n_trials=args.trials)
        return

    from sgan_ids.experiment import run

    run(cfg, STEPS if args.step == "run" else (args.step,))


if __name__ == "__main__":
    main()
