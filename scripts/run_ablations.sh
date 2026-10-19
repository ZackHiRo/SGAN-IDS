#!/usr/bin/env bash
# Ablations on UNSW-NB15: each run changes ONE design choice of the GAN and keeps everything
# else (data, seeds, classifiers, augmentation amount) identical to configs/unsw_nb15.yaml.
#
#   bash scripts/run_ablations.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts

BASE=(--config configs/unsw_nb15.yaml --set 'augment.methods=[none,gan]' --set 'eval.classifiers=[rf,hgb]')
declare -A ABLATIONS=(
  [abl_concat]="gan.conditioning=concat"            # projection critic vs. label concatenation
  [abl_no_ema]="gan.ema_decay=0"                     # EMA of generator weights vs. raw weights
  [abl_alpha0]="gan.class_sampling_alpha=0.0"        # uniform class sampling
  [abl_alpha1]="gan.class_sampling_alpha=1.0"        # empirical class frequencies
  [abl_target2k]="augment.target_per_class=2000"     # less synthetic data
  [abl_target25k]="augment.target_per_class=25000"   # more synthetic data
)
for name in "${!ABLATIONS[@]}"; do
  sgan-ids run "${BASE[@]}" --set "name=${name}" --set "${ABLATIONS[$name]}" 2>&1 | tee "artifacts/${name}.log"
  mkdir -p "results/ablations/${name}"
  cp -r "artifacts/${name}/report.md" "artifacts/${name}/results" "results/ablations/${name}/"
done
