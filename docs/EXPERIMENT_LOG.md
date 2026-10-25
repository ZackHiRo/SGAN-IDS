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

---

## Run 2: CIC-IDS2017, 2026-10-21 (stopped: out of memory)

**Setup:** configs/cic_ids2017.yaml as of commit `bb4d726` (40k GAN steps); 3 seeds;
6 methods; RF, HGB. The full log was not saved; numbers below are from the console.

**What happened:** the 3 generators trained normally (about 53 min each), then the
process was killed during seed 0 while fitting gradient boosting on SMOTE-augmented data.

### What it showed

1. **The rare classes are already almost perfectly detected.** Seed 0, rare-class F1
   (Bot, Web Attack):

   | | RF | HGB |
   |---|---|---|
   | No augmentation | 0.961 | 0.929 |
   | Class weights | 0.965 | 0.968 |
   | Random oversampling | 0.960 | 0.964 |
   | SMOTE | 0.967 | (killed) |

   With so little headroom, CIC-IDS2017 can show whether augmentation *hurts* a
   classifier that already works, but not much about whether it helps.

2. **The GAN's encoding had leaked into the classifiers' input.** The discrete one-hots
   and mode selectors added in `bb4d726` were part of the one encoding everything used,
   so classifiers saw 694 columns instead of about 100. That is what ran out of memory
   (488k rows × 694 columns, which gradient boosting converts to float64), and it made a
   single class-weighted gradient-boosting fit take 16 minutes.

3. **Most of the GAN's progress came before 20k steps.** The best validation MMD reached
   by step 20k was 0.0371 / 0.0344 / 0.0346 (seeds 0 / 1 / 2); by 40k it was
   0.0331 / 0.0336 / 0.0336, a 2–11 % improvement for twice the training time.

### Changes made because of this run
- Separate evaluation and generation encodings (see DECISIONS.md, "Why two encodings").
  Classifiers, SMOTE, oversampling, CTGAN and all metrics are back on the compact
  encoding; only the GAN uses the wide one.
- CIC-IDS2017 back to 20k GAN steps.
- Removed a duplicate copy of run 1's results committed under `sgan-ids-Drive/`.

### Next
Rerun UNSW-NB15 first: it is the dataset where the GAN encoding fix can show an effect.
Then rerun CIC-IDS2017.


---

## Run 3: UNSW-NB15, 2026-10-22

**Setup:** commit `3253ca3` (separate evaluation and generation encodings; GAN with
discrete one-hots and mode selectors, 327 generation-space columns); 3 seeds; 6 methods;
RF, HGB, MLP. 5 h on a Colab GPU.
**Outputs:** [`results/archive/unsw_nb15_run3/`](../results/archive/unsw_nb15_run3/report.md)

The baselines (none, class weights, random oversampling) are identical to run 1 to the
last digit, which confirms that the evaluation space is unchanged.

### What it showed

1. **The encoding fix made our GAN's rows more realistic.**

   | Our GAN | Run 1 | Run 3 |
   |---|---|---|
   | C2ST AUC (0.5 = indistinguishable) | 1.000 | 0.943 |
   | Train on synthetic only, rare-class F1 | 0.004 | 0.173 |
   | Coverage of real rows | 0.578 | 0.603 |

2. **…but augmenting with them got worse.** Rare-class F1 went from 0.212 / 0.281 to
   0.199 / 0.188 (RF / HGB), a mean change of −0.019 vs. no augmentation (4 of 9
   seed × classifier pairs improved).

   Per-class precision and recall show why. Recall rose a lot: Analysis went from 0.19
   to 0.47 (RF) and 0.62 (HGB), the best of any method. But precision collapsed: Worms
   went from 0.52 to 0.19 (RF), against 0.67 with random oversampling, and Shellcode
   from 0.36 to 0.21. The synthetic rare-class rows reach into regions where other
   classes live, so the detector learns to flag real Exploits/DoS flows as rare attacks.
   The fidelity metrics could not show this: they compare synthetic class *c* only with
   real class *c*.

3. **Simple rebalancing still wins.** Class weights: +0.086, 6/6 pairs improved. Random
   oversampling: highest scores (0.311 RF, 0.330 HGB).

4. **The CTGAN baseline looked broken.** A classifier trained on CTGAN's rare-class rows
   detected no real rare flow at all (train-on-synthetic rare-class F1 0.000; sample
   precision 0.06). Suspected cause: `CTGAN.sample(condition_column=…, condition_value=…)`
   only feeds the condition to the generator, and its generated label can differ. Every
   returned row was labelled as the requested class, so rows that CTGAN generated as
   Normal or Exploits were added to the training set as Worms.

## Run 4: CIC-IDS2017, 2026-10-22

**Setup:** commit `3253ca3`; 20k GAN steps; 3 seeds; 6 methods; RF, HGB. 77 evaluation
columns, 694 generation columns. 6.5 h on a Colab GPU.
**Outputs:** [`results/archive/cic_ids2017_run4/`](../results/archive/cic_ids2017_run4/report.md)

### What it showed

1. **No headroom.** With RF, every method scores rare-class F1 0.960–0.966
   (no augmentation: 0.963). HGB starts weaker (0.891) and *any* rebalancing brings it
   to 0.955–0.964, class weights included. Augmentation choice does not matter here.
2. **Our GAN fits this dataset much less well:** C2ST AUC 0.998, precision 0.15,
   coverage 0.05. Its rows alone still teach a classifier rare-class F1 0.758.
3. **CTGAN shows the same zero train-on-synthetic score** as on UNSW-NB15 (0.000;
   sample precision 0.00), supporting the conditional-sampling explanation.

### Changes made because of runs 3 and 4
- **CTGAN rejection sampling:** only rows whose generated label equals the requested
  class are kept, and mismatches are drawn again (as SDV's conditional sampling does).
  The hit rate is recorded per class (`generation.csv`, "Conditional sampling" in the
  report). Synthetic rows cached before this fix are regenerated automatically.
- **Label consistency** for every generator: the share of synthetic rows that a random
  forest trained on real data assigns to their intended class, next to that classifier's
  recall on real test rows. It measures the failure in run 3 directly.

### Next
Rerun both datasets. Our GAN, SMOTE and oversampling come from the cache; only CTGAN
regenerates and the classifiers refit.

---

## Run 5: CIC-IDS2017, 2026-10-25

**Setup:** commit `f51b803` (CTGAN with rejection sampling, label-consistency metric).
Our GAN, SMOTE and oversampling were reused from run 4's cache; CTGAN was refitted.
5.7 h on a Colab GPU.
**Outputs:** [`results/archive/cic_ids2017_run5/`](../results/archive/cic_ids2017_run5/report.md)

### What it showed

1. **The conditional-sampling explanation was right, but only half the story.** Asked
   for Bot, CTGAN generated a Bot-labelled row in 2.2–2.8 % of draws; for Web Attack,
   1.2–1.8 % (`generation.csv`). So in runs 3–4, about 98 % of CTGAN's "rare" rows were
   other traffic carrying a rare label.
2. **Even the label-matched rows are not rare-class traffic.** With rejection sampling
   every CTGAN row has the right label, yet a classifier trained on real data assigns
   **0 %** of them to their class (label consistency 0.00), sample precision is 0.00,
   and train-on-synthetic rare-class F1 is still 0.000. One CTGAN conditioned on the
   label, which it treats as one discrete column among many, does not learn what the
   rare classes look like.
3. **Label consistency exposes our GAN's weakness too:** 61 % of its Bot rows and 53 % of
   its Web Attack rows are classified as their own class, against 96–98 % for real test
   rows, 98 % for SMOTE and 100 % for oversampling. About half of its rare-class rows
   look like other classes.
4. **Detection unchanged:** no headroom (RF 0.960–0.966 for every method).

## Run 6: UNSW-NB15, 2026-10-25 (report not saved)

**Setup:** commit `f51b803`. The run wrote to the Colab machine's local disk instead of
Google Drive (`ARTIFACTS`/`RESULTS` not set), and its outputs were lost with the session.
Numbers below are from the console log. The generators were retrained.

### Rare-class F1, mean ± std over 3 seeds (from the log)

| Method | RF | HGB | MLP |
|---|---|---|---|
| No augmentation | 0.211 | 0.172 | 0.212 |
| Class weights | 0.285 | 0.270 | n/a |
| Random oversampling | 0.311 | 0.330 | 0.147 |
| SMOTE | 0.249 ± 0.001 | 0.252 ± 0.002 | 0.126 ± 0.007 |
| CTGAN (conditional, rejection) | 0.201 ± 0.017 | 0.275 ± 0.007 | 0.185 ± 0.029 |
| cWGAN-GP (ours) | 0.199 ± 0.015 | 0.188 ± 0.007 | 0.149 ± 0.007 |

### What it showed

1. **CTGAN's hit rates are even lower here:** 0.3–0.4 % for Shellcode and 0.05–0.18 %
   for Worms. Even after drawing 200× the requested number, CTGAN produced only
   1,087–3,655 of the 9,892 requested Worms rows (and 5,475–7,683 of 9,073 Shellcode
   rows in two seeds).
2. **Rejection sampling alone moved CTGAN from 0.173 / 0.258 / 0.140 (run 3) to
   0.201 / 0.275 / 0.185**, from removing mislabelled rows. It still does not beat
   simple rebalancing.
3. **Our GAN reproduces run 3** (0.199 / 0.188 / 0.149 vs 0.199 / 0.188 / 0.149) after
   retraining from scratch, so that result is stable.

### Changes made because of runs 5 and 6
- **CTGAN baseline is now per-class** (`augment.ctgan_mode: per_class`, the default):
  one CTGAN fitted on each rare class's own training rows and sampled unconditionally,
  the usual way CTGAN is used for augmentation. Classes with few rows get more epochs so
  that each model gets at least 3,000 generator updates (`augment.ctgan_min_steps`).
  It is also far cheaper: 108–1,548 rows per model instead of ~100k for the joint one.
  The joint conditional version stays available (`ctgan_mode: conditional`).

### Next
Rerun both datasets with `ARTIFACTS` and `RESULTS` set to Google Drive. Only CTGAN is
refitted; everything else comes from the cache.
