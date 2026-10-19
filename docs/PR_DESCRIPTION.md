## v2: Rework SGAN-IDS into a reproducible rare-attack augmentation study

The original thesis code is kept at tag `thesis-v1`. This PR replaces it with a methodologically sound version.

### Why
An audit of the thesis version found problems that invalidate its results:
- **Leakage:** the augmented training set was cross-validated, so synthetic rows ended up in the test folds. The real held-out test set was never used.
- **No rare classes:** `--max-samples` stopped reading after the first CIC-IDS2018 file, so runs held only 2–3 classes and the baseline F1 was 1.000.
- `Timestamp` was one-hot encoded (up to 29k columns) and then compressed with SVD.
- The generator's `tanh` output could not reach the range of the standardised data.
- The "self-attention" had a single head (always weight 1, so effectively a linear layer), and the "projection discriminator" was not a real projection.
- The checkpoint was selected by the generator loss, which does not measure sample quality.

### What changed
- **Package and pipeline:** `src/sgan_ids` with a YAML-driven CLI (`sgan-ids prepare|train|evaluate|report|run|tune`).
- **Data:** UNSW-NB15 (official split) and CIC-IDS2017. Identifier columns removed, rows deduplicated, reversible log + min-max encoding.
- **Model:** conditional WGAN-GP with a projection critic, Gumbel-softmax categorical heads, EMA, and checkpoint selection by validation MMD.
- **Evaluation:** classifiers train on real + synthetic data and are scored on the untouched real test split. Baselines: class weights, random oversampling, SMOTE, CTGAN.
- **Synthetic data quality:** fidelity (MMD, C2ST, PRDC), memorisation (DCR) and train-on-synthetic metrics.
- **Docs:** `docs/DECISIONS.md` justifies every choice and lists each change from the thesis version.
- **Tooling:** 26 tests, ruff, GitHub Actions CI, GPU-pod and ablation scripts.

### Status
- [x] Tests, lint and the end-to-end toy run pass locally (CPU)
- [ ] Full UNSW-NB15 and CIC-IDS2017 runs on GPU (`scripts/run_gpu_pod.sh`)
- [ ] Ablations (`scripts/run_ablations.sh`)
- [ ] Fill the README results table from `results/*/report.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
