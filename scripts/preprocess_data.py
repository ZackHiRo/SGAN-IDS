import argparse
from pathlib import Path
from data.preprocessing import DataForge


def main():
    p = argparse.ArgumentParser(description="Run DataForge pipeline")
    p.add_argument("--data-root", type=Path, default=Path.cwd())
    p.add_argument("--dataset", type=str, default="unified", choices=["nsl_kdd", "cic_ids2017", "unified"], help="Which dataset to preprocess")
    args = p.parse_args()
    if args.dataset == "unified":
        datasets = ["nsl_kdd", "cic_ids2017"]
    else:
        datasets = [args.dataset]
    forge = DataForge(args.data_root).load(datasets)
    (X, _), _ = forge.preprocess()
    print("Preprocessed shape:", X.shape)
    print("Minority classes (<1%):", forge.minority_classes())


if __name__ == "__main__":
    main() 