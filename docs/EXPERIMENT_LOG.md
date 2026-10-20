# Experiment log

What was run, what it showed, and what changed because of it. Full outputs of each
run are in `results/` (archived runs under `results/archive/`).

---

## Run 1: UNSW-NB15, 2026-10-20

**Setup:** configs/unsw_nb15.yaml as of commit `7d4879c`; 3 seeds; 6 methods; RF, HGB, MLP.
**Outputs:** [`results/archive/unsw_nb15_run1/`](../results/archive/unsw_nb15_run1/report.md)

### Detection: rare-class F1 (Analysis, Backdoor, Shellcode, Worms), mean ± std over 3 seeds

| Method | RF | HGB | MLP |
|---|---|---|---|
| No augmentation | 0.211 ± 0.002 | 0.172 ± 0.008 | 0.212 ± 0.017 |
| Class weights | 0.285 ± 0.006 | 0.270 ± 0.005 | n/a |
| Random oversampling | **0.311 ± 0.007** | **0.330 ± 0.005** | 0.147 ± 0.008 |
| SMOTE | 0.247 ± 0.002 | 0.259 ± 0.004 | 0.141 ± 0.011 |
| CTGAN | 0.207 ± 0.002 | 0.257 ± 0.025 | 0.168 ± 0.019 |
| cWGAN-GP (ours) | 0.212 ± 0.011 | 0.281 ± 0.022 | 0.137 ± 0.005 |

### What it showed

1. **Our GAN's samples were trivially separable from real ones** (C2ST AUC 1.000),
   and a classifier trained only on them detected almost no real rare traffic
   (train-on-synthetic rare-class F1 0.004). Fidelity metrics on the continuous features
   were as good as SMOTE's (MMD 0.038 vs 0.036, precision 0.74 vs 0.78).

   *Diagnosis:* the two most important features for telling synthetic from real rows
   were `sttl` and `dttl`, the TTL fields. UNSW-NB15 has 10 numeric columns with at most
   11 distinct values. The generator produced them as continuous numbers that were then
   rounded, which lands on values real traffic never has. Every real Shellcode flow has
   `sttl = 254`; the generator gave 255 in 52 % of rows and 253 in 8 %. Only 30–48 % of
   generated `sttl` values were values the class actually takes, against 98–100 % for
   SMOTE. A classifier learns "TTL 255 means Shellcode", a rule that never fires on real
   traffic.

   *Fix 1:* numeric columns with ≤ 32 distinct training values are now one-hot encoded
   over the observed values and generated through Gumbel-softmax heads
   (`data.max_discrete_values`). The generator can only emit values that occur.

   That fixed the TTLs but not the separability. Single *continuous* features still
   separated synthetic from real rows with AUC 0.88–0.97, against at most 0.73 for SMOTE.
   The cause was exact point masses. In half of real Shellcode flows, jitter, RTT and
   load are exactly 0, and the generator produced exact zeros in 0 % of rows. 45 % of
   Worms flows have `dmean` exactly 45; only 22 % of generated values were values the
   class actually takes.

   *Fix 2:* each continuous column also gets a categorical selector over its frequent
   exact values ("modes": ≥ 5 % of some class and ≥ 10 rows) plus "other", following
   CTAB-GAN's mixed columns. A chosen mode decodes to exactly that value.

2. **Analysis and Backdoor cannot be fixed by augmentation.** They stay below F1 0.12
   with every method. 74 % of Analysis and 77 % of Backdoor training rows have an
   exact duplicate feature vector labelled as another class (`overlap.csv`). The labels
   contradict each other, so the ceiling is in the data, not the model.

3. **Most of the measurable gain is on Worms** (108 training rows). Random oversampling
   took it from 0.216 to 0.618 F1 (RF) and from 0.109 to 0.667 (HGB).

4. **Class weights were the most consistent method** (improved 6/6 seed×classifier
   pairs). Random oversampling gave the highest scores.

5. **Every oversampling method hurt the MLP**, duplication included: rare-class recall
   roughly doubled (0.32 → 0.66) while F1 fell. It predicts the rare classes too often,
   and their precision collapses.

### Changes made because of this run
- Discrete numeric columns encoded as categories, and mode selectors for continuous
  columns (finding 1).
- CIC-IDS2017 label normalisation for every dash variant, and generators that retrain
  when their training data changes (found while starting the CIC-IDS2017 run).

### Check before rerunning
Run 1's splits were re-encoded and the GAN retrained on CPU for 8,000 steps (one seed;
the real runs use 20,000 on GPU). Measured on the same test rows:

| | Run 1 | Fix 1 only | Fix 1 + 2 |
|---|---|---|---|
| Generated `sttl` among the class's real values | 30–48 % | 99.8–100 % | 99.7–100 % |
| Exact zeros in Shellcode `djit`, synthetic vs real | 0 % vs 49 % | not measured | 50 % vs 49 % |
| C2ST AUC (Analysis / Backdoor / Shellcode / Worms) | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 0.98 | 0.98 / 0.98 / 0.93 / 0.90 |
| Train on synthetic only, rare-class F1 (RF) | 0.004 | 0.039 | 0.187 |

Both artefacts are gone and the synthetic rows became usable as training data. They
are still more separable than SMOTE's (0.72–0.81 in run 1); validation MMD was still
improving at 8,000 steps, so the full run will tell how much of that gap is training
time.
