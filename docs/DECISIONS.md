# Design decisions

Why the project is built the way it is. Each section answers a question a reviewer
or interviewer is likely to ask. Section 8 lists what changed since the original
thesis version (git tag `thesis-v1`) and why.

---

## 1. The problem

**What is being solved?**
Intrusion-detection datasets are extremely imbalanced. In UNSW-NB15, *Worms* is 130
of 175,341 training flows (0.07 %). In CIC-IDS2017, *Bot* is about 0.07 % of flows.
A classifier can reach 99 % accuracy while missing these attacks entirely, and rare
attacks are often the most damaging ones. The question is:

> Does adding synthetic flows from a class-conditional GAN improve detection of rare
> attack classes on real, held-out traffic, compared with simpler rebalancing methods?

**Why measure rare-class F1 instead of accuracy?**
Accuracy is dominated by benign traffic. The headline metrics are **macro-F1** (every
class weighs the same) and **rare-class F1/recall**, which is macro-F1 over the
classes that were augmented. Recall matters for an IDS because a missed attack costs
far more than a false alarm. Precision is reported too, because augmentation that
buys recall with a flood of false positives is not useful.

**Why a GAN at all, if SMOTE exists?**
SMOTE creates points on line segments between a rare sample and one of its
neighbours. That works poorly in two ways:

- It cannot produce anything outside the convex hull of the few real samples.
- Interpolating between `proto=tcp` and `proto=udp` yields a record that is half TCP
  and half UDP, which is not a valid flow.

A conditional generator learns the joint distribution of each class, including its
categorical columns. That is a *hypothesis*, not a given: the literature often finds
SMOTE hard to beat on tabular data. So SMOTE, random oversampling, class weights and
CTGAN are all run under exactly the same protocol, and the README reports whichever
method wins.

---

## 2. Data

**Why UNSW-NB15 as the main dataset?**
- It has an **official train/test partition**, so results are comparable with other
  work, and the test set was split off by the dataset authors, not by us.
- It contains modern attack families with several genuinely rare classes: Analysis,
  Backdoor, Shellcode, Worms.
- It has mixed feature types: 3 categorical columns (`proto`, `service`, `state`) and
  about 39 numeric ones. That exercises the categorical output heads.
- At about 257k rows it fits on a laptop.

**Why CIC-IDS2017 as the second dataset?**
It was built by a different lab, with a different feature extractor (CICFlowMeter, 78
flow statistics), on different traffic. If a method only helps on one dataset, that is
a finding about the dataset, not about the method. CIC-IDS2017 is also far more
imbalanced (about 80 % benign).

**Why not NSL-KDD or CIC-IDS2018 (used in the thesis)?**
- NSL-KDD is derived from 1998 DARPA traffic and its weaknesses are well documented.
  It is a legacy benchmark.
- CIC-IDS2018 has about 16 M rows. The thesis loader stopped after the first file to
  save memory, so every run contained a single capture day (*Benign, FTP-BruteForce,
  SSH-Bruteforce*). That means no rare classes and a baseline F1 of 1.000, so there
  was nothing to improve.

**Why not merge datasets into a "unified" one (as in the thesis)?**
NSL-KDD and CIC-IDS2017 have disjoint feature sets and different label taxonomies.
Zero-filling the missing columns creates records that belong to neither dataset.

**Why drop IDs, IPs, ports-as-source and timestamps?**
They identify *a capture*, not *a behaviour*. In CIC datasets each attack ran in a
fixed time window from fixed hosts, so a model can "detect" attacks from the clock or
the IP address with near-perfect accuracy and learn nothing that transfers. In the
thesis, `Timestamp` was even one-hot encoded into up to 29,093 columns. Destination
port is kept (as a categorical) because the targeted service is a legitimate signal.

**Why deduplicate?**
Exact duplicate rows add no information. They inflate the weight of a few flows in
GAN training, and in a random split they put identical rows in both train and test,
which is leakage. For UNSW-NB15, deduplication applies to the *training* partition
only; the official test set is left exactly as published.

**Why cap CIC-IDS2017 at 250,000 rows per class?**
Only BENIGN (about 2.1 M unique flows) and DoS Hulk exceed the cap, so every attack
class is kept whole. Without the cap, 3 seeds × 6 methods × 3 classifiers on more than
2 M rows takes days instead of hours. The cap changes the benign/attack prior. That
mostly affects precision on BENIGN, and it is why the headline metrics are macro and
per-class rather than accuracy.

**Why merge the three web attacks and drop Heartbleed and Infiltration (CIC-IDS2017)?**
- Heartbleed has 11 flows and Infiltration 36. After a 70/15/15 split that leaves
  roughly 7 and 25 training rows, and 2 and 5 test rows. No generative model can learn
  a distribution from 7 points, and an F1 score on 2 test rows is noise.
- Web-attack *SQL Injection* (21 flows) has the same problem. *Brute Force*, *XSS* and
  *SQL Injection* target the same web server and have nearly identical flow
  statistics; they are commonly merged in the literature.

All three choices are in the config (`label_map`, `drop_classes`), so they can be
reverted with one line.

**Known dataset caveats (stated rather than hidden):**
- In UNSW-NB15, the TTL features (`sttl`, `dttl`, `ct_state_ttl`) are near-perfect
  label proxies, an artefact of how the testbed generated attack traffic. They are kept
  by default for comparability with published work. `data.drop_features` removes them.
- CIC-IDS2017 has documented labelling and feature-extraction errors (Engelen et al.,
  2021, *Troubleshooting an intrusion detection dataset: the CICIDS2017 case study*).
  Results on it should be read with that in mind.

---

## 3. Feature encoding

**Why signed `log1p` then min-max to [-1, 1], instead of StandardScaler?**
Flow features are heavy-tailed: byte counts and rates span six or more orders of
magnitude. After standardisation, most values sit near 0 and a few lie hundreds of
standard deviations away.

The thesis generator ended in `tanh` (range [-1, 1]), so it could not produce the
real data at all. The log compresses the tails, and min-max scaling with training
statistics then puts every numeric column exactly in the range of `tanh`. The
transform is monotone (tree classifiers are unaffected) and exactly invertible.

The log is applied only to columns whose magnitude exceeds 100; flags and small
counts are left linear.

**Why one-hot categoricals with a Gumbel-softmax output?**
Real categorical blocks are exact one-hots. A plain softmax output is never exactly
0/1, so the critic can tell real from fake just by checking that, and training
collapses into fighting that artefact. The Gumbel-softmax with a low temperature
(0.2), as used in CTGAN, gives near-one-hot outputs that remain differentiable. At
sampling time the output is a hard one-hot.

Categories rarer than 0.1 % of training rows (most of UNSW's 130+ `proto` values) are
grouped into one "infrequent" category, which keeps the encoding compact.

**Why "project" synthetic rows (round integers, snap categoricals, clip to range)?**
A synthetic flow with 3.4 packets is not a valid flow. Every generated or interpolated
row (GAN and SMOTE alike) is decoded to raw units, made valid and re-encoded, so
classifiers only ever see records that could exist.

**Why not CTGAN's mode-specific normalisation (a Gaussian mixture per column)?**
It handles multi-modal columns better but adds a fitted mixture model per column and a
more complex output head. The log + min-max scheme is simpler, exactly invertible, and
shared by every method. CTGAN itself, with its own normalisation, is included as a
baseline, so the comparison still covers that approach.

---

## 4. The model

**Why WGAN-GP (Gulrajani et al., 2017)?**
- The standard GAN loss saturates when the critic wins, which happens easily on
  low-dimensional tabular data.
- The Wasserstein loss with a gradient penalty gives useful gradients throughout and is
  the de facto choice for tabular GANs (CTGAN uses it).
- Defaults come from the paper: 5 critic steps per generator step, λ = 10, Adam with
  lr 1e-4 and betas (0.5, 0.9).

**Why a projection critic instead of an auxiliary classifier (AC-GAN)?**
An AC-GAN generator is rewarded for samples that are *easy to classify*. Known
results show this pushes samples away from class boundaries and reduces diversity
(Shu et al., 2017; Gong et al., 2019). For augmentation that is exactly wrong: the
hard, boundary-near samples are the useful ones.

The projection critic (Miyato & Koyama, 2018), `D(x, y) = wᵀφ(x) + embed(y)ᵀφ(x)`,
asks a single question: "is this a realistic member of class *y*?" It is the
conditioning used by modern conditional GANs (e.g. BigGAN). Label concatenation is
kept as an ablation.

**Why no batch norm, spectral norm, minibatch discrimination, self-attention or feature
matching (all in the thesis)?**
- **Batch norm / minibatch discrimination in the critic:** the gradient penalty
  assumes the critic scores each sample independently. Both make a sample's score
  depend on the rest of the batch.
- **Spectral norm** enforces a Lipschitz bound, which is already the gradient
  penalty's job. Using both over-constrains the critic.
- **Self-attention:** the thesis version always ran with one head, and a softmax over a
  single element is always 1, so the "attention" was a linear layer. Real attention
  over features is a reasonable research direction, but it should only come back if an
  ablation shows it helps.
- **Feature matching** was a fix for the original GAN loss. With WGAN-GP it is one
  more loss weight to tune, with no clear benefit.

The principle: every component must earn its place in an ablation
(`scripts/run_ablations.sh`). Stacking seven stabilisation tricks makes it impossible
to say which one does anything.

**Why residual MLP blocks?**
On tabular data, MLPs are the standard and strong choice; there is no spatial or
sequential structure for convolutions or transformers to exploit. Residual
connections with LayerNorm make a 3-layer MLP train reliably without further tuning.

**Why EMA of the generator weights?**
GAN weights oscillate. Averaging them (decay 0.999, with warm-up so the random
initialisation does not dominate early on) gives consistently better samples at no
training cost (Yazıcı et al., 2019). An ablation (`ema_decay=0`) tests the claim.

**Why sample classes with p(c) ∝ n_c^0.5?**
- Empirical frequencies (α = 1): the generator almost never sees Worms.
- Uniform (α = 0): the 130 Worms rows are replayed as often as 40,000 Generic rows,
  which invites memorisation.
- The square root is the middle ground (CTGAN uses a log-frequency variant).

Real and fake batches use the *same* labels, so the critic always compares
p(x | y) with p_G(x | y). Every critic step draws a fresh real batch; the thesis
reused one batch for all five critic steps.

**How is the checkpoint chosen?**
Every 1,000 steps, the EMA generator is scored on the **validation** split by
class-averaged MMD between generated and real validation rows. The best checkpoint is
kept.

- The thesis used the generator loss. In a WGAN that value depends on the current
  critic, so it is not comparable across checkpoints; it is not a quality measure.
- The MMD uses one fixed RBF bandwidth (median heuristic, computed once), so scores
  are comparable across steps.
- The average is over classes, so Worms counts as much as Generic.

**How were hyperparameters chosen?**
The paper defaults above. `sgan-ids tune` runs an Optuna search over learning rates,
critic steps, λ, width, latent size and α, minimising the same validation MMD. Tuning
never sees the test split.

---

## 5. Augmentation protocol

**How much synthetic data?**
Every class with fewer than 10,000 training rows is topped up to 10,000. That is about
the size of UNSW's mid-sized classes (Reconnaissance, DoS). The rare classes become
comparable to them without outweighing the large classes. The same plan is given to
every method, so they differ only in *where* the extra rows come from. The ablation
script also runs 2,000 and 25,000.

**Baselines and why each one is there:**

| Method | Why it is in the comparison |
|---|---|
| No augmentation | Reference point. |
| Class weights | Rebalancing without any new data, the cheapest possible fix. |
| Random oversampling | New rows that are exact copies. Shows what "more rows" alone does, and anchors the memorisation metric at 0. |
| SMOTE | The standard method every reviewer will ask about. |
| CTGAN | The standard *deep* tabular generator. Shows whether this project's GAN adds anything over an off-the-shelf one. |

---

## 6. Evaluation protocol

**Train on (real + synthetic) train, test on the real test split. Nothing else.**
In the thesis, the augmented set was cross-validated, so synthetic rows ended up in
the test folds. Synthetic rows are typically easier to classify than real ones, which
inflates the augmented score. The held-out real test set was never used. Now:

- Synthetic rows are only ever added to training data.
- The GAN, SMOTE and CTGAN are fitted on the training split only.
- Hyperparameters and checkpoints are chosen on validation.
- The test split is touched once, for the final scores.

**Why three classifiers (Random Forest, HistGradientBoosting, MLP)?**
They have different inductive biases. Tree ensembles are the strongest standard IDS
baselines and are insensitive to monotone feature scaling; an MLP responds very
differently to added data. A method that helps only one of them is a weaker result,
and the report shows each separately.

**Why seeds and paired differences instead of a Wilcoxon test over 5 CV folds?**
With 5 paired values, a one-sided Wilcoxon test cannot produce p < 0.03, and the
thesis folds were contaminated anyway. Each seed retrains the GAN, regenerates the
data and refits the classifier, so the reported mean ± std covers the full randomness
of the pipeline.

The report also shows:

- The change in rare-class F1 vs. no augmentation, **paired** by seed and classifier,
  with how many of the pairs improved.
- A 95 % bootstrap interval over test rows, which reflects how few test examples the
  rare classes have.

**What do the synthetic-data metrics tell you?**
All are computed per augmented class against real **test** rows (not the validation
rows used for checkpoint selection), then averaged.

| Metric | Question it answers | Ideal |
|---|---|---|
| C2ST AUC | Can a random forest tell synthetic from real? | 0.5 |
| MMD | Kernel distance between the two distributions | 0 |
| Precision / density | Do synthetic rows fall where real rows are? (realism) | 1 |
| Recall / coverage | Does the synthetic data cover all of the real data? (diversity, detects mode collapse) | 1 |
| Marginal Wasserstein, correlation gap | Are per-feature distributions and pairwise correlations preserved? | 0 |
| DCR ratio | Are synthetic rows as far from the training set as genuinely new real traffic is, or are they copies? | ≈ 1 |
| Train-on-synthetic | With the real rows of a rare class *removed*, how well does a classifier trained on its synthetic rows detect it? Pure utility of the generator. | high |

A good augmenter must do well on realism, diversity *and* novelty. Random oversampling
gets perfect realism and zero novelty; a collapsed GAN gets high precision and low
coverage. Looking at one metric alone is how GAN papers mislead.

---

## 7. Limitations

- The data is **flow features**, not packets. A synthetic flow is a plausible feature
  vector, not replayable traffic.
- Two benchmark datasets with known artefacts (section 2). Performance on a real
  network would need validation on its own traffic.
- One GAN architecture family, three seeds per configuration, one augmentation amount
  in the main table (more in the ablations).
- **About the name:** "StealthGAN" was the thesis title. The project generates rare
  attack samples to *train detectors*; it does not attempt evasion attacks against an
  IDS.

---

## 8. Changes since the thesis version (`thesis-v1`)

| Thesis version | Problem | Now |
|---|---|---|
| Augmented data cross-validated with synthetic rows in the test folds; real test set unused | Leakage, inflated uplift | Train on augmented train, score on the untouched real test split |
| `--max-samples` read files in order and stopped early | CIC-IDS2018 runs held only one day (2 to 3 classes, baseline F1 = 1.000) | Official / stratified splits; benign capped per class, attacks kept whole |
| `Timestamp` one-hot encoded (up to 29k columns), then TruncatedSVD to 256 | Leakage, and the GAN modelled an SVD space | Identifier columns dropped; GAN works on the real features |
| "Unified" NSL-KDD + CIC-IDS2017 with zero-filling | Meaningless combined data | Two datasets evaluated separately |
| StandardScaler data, `tanh` generator output | Generator could not reach the data range | Signed log + min-max to [-1, 1] |
| One-head "self-attention" | Softmax over one element = 1; it was a linear layer | Removed |
| "Projection discriminator" adding the embedding | Not a projection | Real inner-product projection critic |
| Minibatch discrimination + spectral norm + GP + feature matching + AC head | Conflicting regularisers, no ablation | WGAN-GP + projection + EMA, each ablated |
| Best checkpoint = lowest validation generator loss | Not a quality measure in a WGAN; critic loss was diverging (about -650) | Validation class-averaged MMD |
| 5 critic steps on the same real batch | Critic overfits one batch | Fresh real batch each step |
| Optuna objective computed on 2-class data where baseline F1 was already 1.000; tuned values never used by evaluation | No signal to tune on (one 30-trial study ended with best objective 0.0) | Optuna on validation MMD, through the same config |
| "KL/JS divergence" between normalised feature means | Not a distribution divergence | MMD, C2ST, PRDC, Wasserstein, correlation gap |
| No baselines | No way to tell if the GAN helps | Class weights, ROS, SMOTE, CTGAN |
| No memorisation check | Rare classes replayed thousands of times | DCR ratio and exact-copy rate |
| No tests, broken preprocessing script, unused configs | | Package with CLI, YAML configs that drive every run, 23 tests, CI |
