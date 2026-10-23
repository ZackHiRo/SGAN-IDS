# Results: unsw_nb15

Dataset `unsw_nb15`, split: official. Rows: {'train': 91579, 'val': 16161, 'test': 82332}. Encoded features: 60 (GAN generation space: 327).

Augmented classes (rows added): {'Analysis': 8645, 'Backdoor': 8695, 'Shellcode': 9073, 'Worms': 9892}

Seeds: [0, 1, 2]. Values are mean ± std over seeds; all scores are on the real, untouched test split.

## Detection (train on real + synthetic, test on real)

| classifier | method | macro_f1 | rare_macro_f1 | rare_macro_recall | balanced_accuracy |
|---|---|---|---|---|---|
| rf | none | 0.477 ± 0.003 | 0.211 ± 0.002 | 0.321 ± 0.003 | 0.525 ± 0.000 |
| rf | class_weight | 0.508 ± 0.003 | 0.285 ± 0.006 | 0.468 ± 0.015 | 0.572 ± 0.005 |
| rf | ros | 0.518 ± 0.003 | 0.311 ± 0.007 | 0.571 ± 0.009 | 0.611 ± 0.003 |
| rf | smote | 0.491 ± 0.000 | 0.249 ± 0.001 | 0.620 ± 0.005 | 0.626 ± 0.001 |
| rf | ctgan | 0.460 ± 0.007 | 0.173 ± 0.013 | 0.335 ± 0.014 | 0.527 ± 0.006 |
| rf | gan | 0.473 ± 0.005 | 0.199 ± 0.015 | 0.525 ± 0.030 | 0.592 ± 0.011 |
| hgb | none | 0.453 ± 0.010 | 0.172 ± 0.008 | 0.369 ± 0.021 | 0.545 ± 0.012 |
| hgb | class_weight | 0.488 ± 0.003 | 0.270 ± 0.005 | 0.695 ± 0.003 | 0.654 ± 0.001 |
| hgb | ros | 0.526 ± 0.002 | 0.330 ± 0.005 | 0.606 ± 0.015 | 0.625 ± 0.005 |
| hgb | smote | 0.493 ± 0.001 | 0.252 ± 0.002 | 0.606 ± 0.011 | 0.622 ± 0.005 |
| hgb | ctgan | 0.499 ± 0.009 | 0.258 ± 0.025 | 0.396 ± 0.040 | 0.558 ± 0.014 |
| hgb | gan | 0.465 ± 0.005 | 0.188 ± 0.007 | 0.548 ± 0.008 | 0.599 ± 0.008 |
| mlp | none | 0.474 ± 0.031 | 0.212 ± 0.017 | 0.317 ± 0.038 | 0.523 ± 0.019 |
| mlp | ros | 0.437 ± 0.001 | 0.147 ± 0.008 | 0.665 ± 0.014 | 0.632 ± 0.005 |
| mlp | smote | 0.429 ± 0.003 | 0.126 ± 0.007 | 0.664 ± 0.007 | 0.632 ± 0.004 |
| mlp | ctgan | 0.443 ± 0.017 | 0.140 ± 0.014 | 0.256 ± 0.074 | 0.503 ± 0.019 |
| mlp | gan | 0.434 ± 0.011 | 0.149 ± 0.007 | 0.589 ± 0.007 | 0.599 ± 0.013 |

### Rare-class F1 change vs. no augmentation (paired by seed and classifier)

| method | mean Δ rare-F1 | seeds×classifiers improved |
|---|---|---|
| class_weight | +0.086 | 6/6 |
| ros | +0.065 | 6/9 |
| smote | +0.011 | 6/9 |
| ctgan | -0.008 | 3/9 |
| gan | -0.019 | 4/9 |

### Per-class F1 on augmented classes (rf)

| method | Analysis | Backdoor | Shellcode | Worms |
|---|---|---|---|---|
| No augmentation | 0.084 | 0.074 | 0.469 | 0.216 |
| Class weights | 0.105 | 0.106 | 0.468 | 0.461 |
| Random oversampling | 0.108 | 0.107 | 0.412 | 0.618 |
| SMOTE | 0.082 | 0.117 | 0.321 | 0.476 |
| CTGAN | 0.082 | 0.088 | 0.428 | 0.093 |
| Ours (cWGAN-GP) | 0.124 | 0.092 | 0.333 | 0.248 |

### Per-class F1 on augmented classes (hgb)

| method | Analysis | Backdoor | Shellcode | Worms |
|---|---|---|---|---|
| No augmentation | 0.048 | 0.085 | 0.446 | 0.109 |
| Class weights | 0.088 | 0.109 | 0.290 | 0.594 |
| Random oversampling | 0.110 | 0.099 | 0.446 | 0.667 |
| SMOTE | 0.086 | 0.107 | 0.362 | 0.452 |
| CTGAN | 0.090 | 0.093 | 0.478 | 0.371 |
| Ours (cWGAN-GP) | 0.135 | 0.072 | 0.368 | 0.177 |

### Per-class F1 on augmented classes (mlp)

| method | Analysis | Backdoor | Shellcode | Worms |
|---|---|---|---|---|
| No augmentation | 0.083 | 0.119 | 0.449 | 0.198 |
| Random oversampling | 0.067 | 0.117 | 0.251 | 0.151 |
| SMOTE | 0.082 | 0.108 | 0.214 | 0.099 |
| CTGAN | 0.085 | 0.062 | 0.348 | 0.067 |
| Ours (cWGAN-GP) | 0.118 | 0.093 | 0.276 | 0.111 |

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
| ctgan | 1.000 ± 0.000 | 0.343 ± 0.004 | 0.058 ± 0.004 | 0.946 ± 0.044 | 0.101 ± 0.007 | 0.228 ± 0.003 | 0.194 ± 0.002 | 36.413 ± 0.391 | 0.000 ± 0.000 |
| gan | 0.943 ± 0.003 | 0.038 ± 0.001 | 0.717 ± 0.007 | 0.618 ± 0.038 | 0.603 ± 0.008 | 0.048 ± 0.002 | 0.069 ± 0.001 | 1.789 ± 0.002 | 0.048 ± 0.001 |

![Fidelity vs privacy](figures/fidelity_vs_privacy.png)

## Train on synthetic only (real rows of augmented classes removed, RF)

| method | rare_macro_f1 | rare_macro_recall | macro_f1 |
|---|---|---|---|
| ros | 0.313 ± 0.009 | 0.568 ± 0.011 | 0.519 ± 0.004 |
| smote | 0.248 ± 0.001 | 0.595 ± 0.006 | 0.489 ± 0.001 |
| ctgan | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.397 ± 0.001 |
| gan | 0.173 ± 0.011 | 0.385 ± 0.011 | 0.466 ± 0.006 |

## GAN training

![Training curves](figures/training.png)
