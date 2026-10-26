# Results: unsw_nb15

Dataset `unsw_nb15`, split: official. Rows: {'train': 91579, 'val': 16161, 'test': 82332}. Encoded features: 60 (GAN generation space: 327).

Augmented classes (rows added): {'Analysis': 8645, 'Backdoor': 8695, 'Shellcode': 9073, 'Worms': 9892}

Seeds: [0, 1, 2]. Values are mean ± std over seeds; all scores are on the real, untouched test split.

## Detection (train on real + synthetic, test on real)

| classifier | method | macro_f1 | rare_macro_f1 | rare_macro_recall | balanced_accuracy |
|---|---|---|---|---|---|
| rf | none | 0.477 ± 0.003 | 0.211 ± 0.002 | 0.321 ± 0.003 | 0.525 ± 0.000 |
| rf | class_weight | 0.507 ± 0.004 | 0.282 ± 0.009 | 0.481 ± 0.008 | 0.578 ± 0.003 |
| rf | balanced_weight | 0.508 ± 0.003 | 0.285 ± 0.006 | 0.468 ± 0.015 | 0.572 ± 0.005 |
| rf | ros | 0.518 ± 0.003 | 0.311 ± 0.007 | 0.571 ± 0.009 | 0.611 ± 0.003 |
| rf | smote | 0.491 ± 0.000 | 0.249 ± 0.001 | 0.620 ± 0.005 | 0.626 ± 0.001 |
| rf | ctgan | 0.476 ± 0.004 | 0.208 ± 0.007 | 0.383 ± 0.003 | 0.545 ± 0.001 |
| rf | gan | 0.476 ± 0.005 | 0.209 ± 0.012 | 0.533 ± 0.024 | 0.594 ± 0.009 |
| hgb | none | 0.453 ± 0.010 | 0.172 ± 0.008 | 0.369 ± 0.021 | 0.545 ± 0.012 |
| hgb | class_weight | 0.504 ± 0.007 | 0.280 ± 0.014 | 0.684 ± 0.010 | 0.652 ± 0.003 |
| hgb | balanced_weight | 0.488 ± 0.003 | 0.270 ± 0.005 | 0.695 ± 0.003 | 0.654 ± 0.001 |
| hgb | ros | 0.526 ± 0.002 | 0.330 ± 0.005 | 0.606 ± 0.015 | 0.625 ± 0.005 |
| hgb | smote | 0.493 ± 0.001 | 0.252 ± 0.002 | 0.606 ± 0.011 | 0.622 ± 0.005 |
| hgb | ctgan | 0.491 ± 0.030 | 0.248 ± 0.055 | 0.496 ± 0.024 | 0.587 ± 0.018 |
| hgb | gan | 0.493 ± 0.020 | 0.247 ± 0.052 | 0.542 ± 0.014 | 0.601 ± 0.006 |
| mlp | none | 0.474 ± 0.031 | 0.212 ± 0.017 | 0.317 ± 0.038 | 0.523 ± 0.019 |
| mlp | ros | 0.437 ± 0.001 | 0.147 ± 0.008 | 0.665 ± 0.014 | 0.632 ± 0.005 |
| mlp | smote | 0.429 ± 0.003 | 0.126 ± 0.007 | 0.664 ± 0.007 | 0.632 ± 0.004 |
| mlp | ctgan | 0.459 ± 0.037 | 0.188 ± 0.051 | 0.243 ± 0.135 | 0.499 ± 0.054 |
| mlp | gan | 0.445 ± 0.015 | 0.162 ± 0.020 | 0.551 ± 0.041 | 0.592 ± 0.005 |

### Rare-class F1 change vs. no augmentation (paired by seed and classifier)

| method | mean Δ rare-F1 | seeds×classifiers improved |
|---|---|---|
| class_weight | +0.090 | 6/6 |
| balanced_weight | +0.086 | 6/6 |
| ros | +0.065 | 6/9 |
| smote | +0.011 | 6/9 |
| ctgan | +0.016 | 5/9 |
| gan | +0.007 | 4/9 |

### Per-class F1 on augmented classes (rf)

| method | Analysis | Backdoor | Shellcode | Worms |
|---|---|---|---|---|
| No augmentation | 0.084 | 0.074 | 0.469 | 0.216 |
| Class weights (rare classes) | 0.121 | 0.105 | 0.458 | 0.444 |
| Balanced class weights | 0.105 | 0.106 | 0.468 | 0.461 |
| Random oversampling | 0.108 | 0.107 | 0.412 | 0.618 |
| SMOTE | 0.082 | 0.117 | 0.321 | 0.476 |
| CTGAN | 0.078 | 0.106 | 0.428 | 0.218 |
| Ours (cWGAN-GP) | 0.119 | 0.097 | 0.332 | 0.287 |

### Per-class F1 on augmented classes (hgb)

| method | Analysis | Backdoor | Shellcode | Worms |
|---|---|---|---|---|
| No augmentation | 0.048 | 0.085 | 0.446 | 0.109 |
| Class weights (rare classes) | 0.091 | 0.107 | 0.317 | 0.607 |
| Balanced class weights | 0.088 | 0.109 | 0.290 | 0.594 |
| Random oversampling | 0.110 | 0.099 | 0.446 | 0.667 |
| SMOTE | 0.086 | 0.107 | 0.362 | 0.452 |
| CTGAN | 0.054 | 0.137 | 0.439 | 0.362 |
| Ours (cWGAN-GP) | 0.134 | 0.075 | 0.388 | 0.392 |

### Per-class F1 on augmented classes (mlp)

| method | Analysis | Backdoor | Shellcode | Worms |
|---|---|---|---|---|
| No augmentation | 0.083 | 0.119 | 0.449 | 0.198 |
| Random oversampling | 0.067 | 0.117 | 0.251 | 0.151 |
| SMOTE | 0.082 | 0.108 | 0.214 | 0.099 |
| CTGAN | 0.046 | 0.108 | 0.392 | 0.207 |
| Ours (cWGAN-GP) | 0.122 | 0.087 | 0.295 | 0.143 |

![Per-class F1](figures/per_class_f1.png)

### How separable are the rare classes?

Share of each rare class's test rows whose exact feature vector also occurs in the training data under another label. No classifier can be sure of those rows, whatever augmentation is used.

| class | n_test | test_seen_with_other_label | test_seen_with_same_label_only | test_unseen | train_shared_with_other_label |
|---|---|---|---|---|---|
| Analysis | 677 | 17.1 % | 0.0 % | 82.9 % | 73.9 % |
| Backdoor | 583 | 1.4 % | 0.0 % | 98.6 % | 76.6 % |
| Shellcode | 378 | 0.3 % | 2.9 % | 96.8 % | 0.5 % |
| Worms | 44 | 0.0 % | 0.0 % | 100.0 % | 7.4 % |

## Synthetic data quality (augmented classes, vs. real test rows)

C2ST AUC: 0.5 = indistinguishable from real. DCR ratio: ~1 = as novel as real held-out traffic, 0 = copies of training rows.

| method | c2st_auc | mmd | precision | recall | coverage | wasserstein | correlation_gap | dcr_ratio | exact_copy_rate |
|---|---|---|---|---|---|---|---|---|---|
| ros | 0.805 ± 0.006 | 0.037 ± 0.001 | 0.759 ± 0.003 | 0.320 ± 0.005 | 0.631 ± 0.002 | 0.048 ± 0.002 | 0.059 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| smote | 0.811 ± 0.011 | 0.036 ± 0.003 | 0.780 ± 0.008 | 0.319 ± 0.018 | 0.633 ± 0.002 | 0.045 ± 0.003 | 0.058 ± 0.000 | 0.226 ± 0.004 | 0.146 ± 0.005 |
| ctgan | 1.000 ± 0.000 | 0.305 ± 0.011 | 0.012 ± 0.005 | 0.804 ± 0.042 | 0.065 ± 0.016 | 0.138 ± 0.008 | 0.123 ± 0.003 | 30.385 ± 0.891 | 0.000 ± 0.000 |
| gan | 0.944 ± 0.010 | 0.039 ± 0.001 | 0.710 ± 0.006 | 0.642 ± 0.043 | 0.598 ± 0.017 | 0.048 ± 0.001 | 0.070 ± 0.002 | 1.838 ± 0.051 | 0.045 ± 0.005 |

![Fidelity vs privacy](figures/fidelity_vs_privacy.png)

### Do synthetic rows look like their own class?

Share of synthetic rows that a random forest trained on real training data assigns to the intended class (mean over seeds). Compare each method with the last row, the same classifier's recall on real *held-out* rows: that is what genuinely new traffic of the class scores. Oversampling and SMOTE score high partly because their rows are copies of, or interpolations between, that classifier's own training rows; it does not mean they are better samples.

| method | Analysis | Backdoor | Shellcode | Worms |
|---|---|---|---|---|
| ros | 0.42 | 0.41 | 1.00 | 0.95 |
| smote | 0.33 | 0.38 | 0.87 | 0.55 |
| ctgan | 0.02 | 0.00 | 0.42 | 0.00 |
| gan | 0.13 | 0.23 | 0.61 | 0.02 |
| *real test rows* | 0.16 | 0.31 | 0.70 | 0.14 |

### Generator details

Rows requested and generated per class. Conditional CTGAN: share of draws whose label matched the requested class (mismatches are rejected). Per-class CTGAN: training rows and epochs of each class's model.

| method | class | requested | generated | fit_rows | epochs |
|---|---|---|---|---|---|
| ctgan | Analysis | 8645.0 | 8645.0 | 1355.0 | 1500.0 |
| ctgan | Backdoor | 8695.0 | 8695.0 | 1305.0 | 1500.0 |
| ctgan | Shellcode | 9073.0 | 9073.0 | 927.0 | 3000.0 |
| ctgan | Worms | 9892.0 | 9892.0 | 108.0 | 3000.0 |

## Train on synthetic only (real rows of augmented classes removed, RF)

| method | rare_macro_f1 | rare_macro_recall | macro_f1 |
|---|---|---|---|
| ros | 0.313 ± 0.009 | 0.568 ± 0.011 | 0.519 ± 0.004 |
| smote | 0.248 ± 0.001 | 0.595 ± 0.006 | 0.489 ± 0.001 |
| ctgan | 0.003 ± 0.001 | 0.001 ± 0.001 | 0.398 ± 0.003 |
| gan | 0.175 ± 0.005 | 0.380 ± 0.024 | 0.467 ± 0.003 |

## GAN training

![Training curves](figures/training.png)
