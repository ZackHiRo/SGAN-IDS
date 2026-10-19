#!/usr/bin/env bash
# Full experiment on a GPU machine (pod or Colab).
#
#   bash scripts/run_gpu_pod.sh                # both datasets
#   bash scripts/run_gpu_pod.sh unsw_nb15      # one dataset
#
# Measured on a Colab GPU for UNSW-NB15 (3 seeds): about 4 h, of which CTGAN took
# 2.5 h before its training rows were capped (augment.ctgan_max_rows_per_class).
#
# Where outputs go (set these to a Google Drive folder on Colab, so nothing is lost
# when the session ends; re-running then reuses trained generators and cached
# synthetic data):
#   ARTIFACTS  full outputs (splits, generators, synthetic data)   default: artifacts
#   RESULTS    small, committable copies (reports, CSVs, figures)  default: results
set -euo pipefail

cd "$(dirname "$0")/.."
ARTIFACTS="${ARTIFACTS:-artifacts}"
RESULTS="${RESULTS:-results}"
pip install -q -e ".[ctgan,tune]"

# 1. Seconds-long sanity check of the whole pipeline before spending GPU hours.
sgan-ids run --config configs/toy.yaml --set "output_dir=${ARTIFACTS}" \
  --set gan.n_steps=200 --set gan.eval_every=100

if [ $# -gt 0 ]; then DATASETS=("$@"); else DATASETS=(unsw_nb15 cic_ids2017); fi
for ds in "${DATASETS[@]}"; do
  out="${ARTIFACTS}/${ds}"
  # 2. Prepare once and print the class counts, so problems show up before training.
  sgan-ids prepare --config "configs/${ds}.yaml" --set "output_dir=${ARTIFACTS}"
  python -c "import json; r=json.load(open('${out}/data/data_report.json')); print(json.dumps(r['class_counts'], indent=1))"

  # 3. Train one GAN per seed, augment, evaluate, write ${out}/report.md.
  sgan-ids run --config "configs/${ds}.yaml" --set "output_dir=${ARTIFACTS}" 2>&1 | tee "${out}/run.log"

  # 4. Copy the small, human-readable outputs so they can be committed.
  mkdir -p "${RESULTS}/${ds}"
  cp -r "${out}/report.md" "${out}/figures" "${out}/results" "${out}/config.yaml" \
        "${out}/data/data_report.json" "${out}/run.log" "${RESULTS}/${ds}/"
  for s in "${out}"/gan/seed_*; do cp "$s/history.csv" "${RESULTS}/${ds}/history_$(basename "$s").csv"; done
done
echo "Done. Results in ${RESULTS}/ (commit it, or zip it and send it back)."
