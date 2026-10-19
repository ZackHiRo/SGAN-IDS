#!/usr/bin/env bash
# Full experiment on a GPU pod. Rough wall-time estimate on one modern GPU (not measured yet):
# a few hours per dataset; CTGAN and the CPU classifiers dominate, not our GAN.
#
#   bash scripts/run_gpu_pod.sh            # both datasets
#   bash scripts/run_gpu_pod.sh unsw_nb15  # one dataset
set -euo pipefail

cd "$(dirname "$0")/.."
pip install -q -e ".[ctgan,tune]"

# 1. Seconds-long sanity check of the whole pipeline before spending GPU hours.
sgan-ids run --config configs/toy.yaml --set gan.n_steps=200 --set gan.eval_every=100

if [ $# -gt 0 ]; then DATASETS=("$@"); else DATASETS=(unsw_nb15 cic_ids2017); fi
for ds in "${DATASETS[@]}"; do
  # 2. Prepare once and print the class counts, so problems show up before training.
  sgan-ids prepare --config "configs/${ds}.yaml"
  python -c "import json; r=json.load(open('artifacts/${ds}/data/data_report.json')); print(json.dumps(r['class_counts'], indent=1))"

  # 3. Train one GAN per seed, augment, evaluate, write artifacts/${ds}/report.md.
  sgan-ids run --config "configs/${ds}.yaml" 2>&1 | tee "artifacts/${ds}/run.log"
done

# 4. Copy the small, human-readable outputs into results/ so they can be committed.
for ds in "${DATASETS[@]}"; do
  mkdir -p "results/${ds}"
  cp -r "artifacts/${ds}/report.md" "artifacts/${ds}/figures" "artifacts/${ds}/results" \
        "artifacts/${ds}/config.yaml" "artifacts/${ds}/data/data_report.json" "results/${ds}/"
  for s in artifacts/${ds}/gan/seed_*; do cp "$s/history.csv" "results/${ds}/history_$(basename "$s").csv"; done
done
echo "Done. Commit the results/ folder (tar it and send it back if you prefer)."
