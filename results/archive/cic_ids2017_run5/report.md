# Results: cic_ids2017

Dataset `cic_ids2017`, split: stratified test_size=0.15. Rows: {'train': 488188, 'val': 86151, 'test': 101355}. Encoded features: 77 (GAN generation space: 694).

Augmented classes (rows added): {'Bot': 8592, 'Web Attack': 8452}

Seeds: [0, 1, 2]. Values are mean ± std over seeds; all scores are on the real, untouched test split.

## Detection (train on real + synthetic, test on real)

| classifier | method | macro_f1 | rare_macro_f1 | rare_macro_recall | balanced_accuracy |
|---|---|---|---|---|---|
| rf | none | 0.992 ± 0.000 | 0.963 ± 0.002 | 0.968 ± 0.002 | 0.993 ± 0.000 |
| rf | class_weight | 0.993 ± 0.000 | 0.965 ± 0.002 | 0.975 ± 0.002 | 0.994 ± 0.001 |
| rf | ros | 0.991 ± 0.001 | 0.960 ± 0.003 | 0.978 ± 0.005 | 0.994 ± 0.001 |
| rf | smote | 0.993 ± 0.000 | 0.966 ± 0.002 | 0.984 ± 0.001 | 0.996 ± 0.000 |
| rf | ctgan | 0.992 ± 0.000 | 0.964 ± 0.001 | 0.973 ± 0.003 | 0.994 ± 0.001 |
| rf | gan | 0.992 ± 0.000 | 0.965 ± 0.001 | 0.977 ± 0.003 | 0.994 ± 0.000 |
| hgb | none | 0.970 ± 0.006 | 0.891 ± 0.024 | 0.909 ± 0.057 | 0.975 ± 0.012 |
| hgb | class_weight | 0.991 ± 0.000 | 0.959 ± 0.001 | 0.993 ± 0.001 | 0.998 ± 0.000 |
| hgb | ros | 0.987 ± 0.002 | 0.964 ± 0.003 | 0.984 ± 0.001 | 0.992 ± 0.002 |
| hgb | smote | 0.985 ± 0.004 | 0.964 ± 0.003 | 0.986 ± 0.002 | 0.992 ± 0.001 |
| hgb | ctgan | 0.985 ± 0.001 | 0.946 ± 0.004 | 0.964 ± 0.012 | 0.989 ± 0.003 |
| hgb | gan | 0.982 ± 0.002 | 0.955 ± 0.003 | 0.981 ± 0.003 | 0.988 ± 0.002 |

### Rare-class F1 change vs. no augmentation (paired by seed and classifier)

| method | mean Δ rare-F1 | seeds×classifiers improved |
|---|---|---|
| class_weight | +0.035 | 6/6 |
| ros | +0.035 | 3/6 |
| smote | +0.038 | 6/6 |
| ctgan | +0.028 | 5/6 |
| gan | +0.033 | 6/6 |

### Per-class F1 on augmented classes (rf)

| method | Bot | Web Attack |
|---|---|---|
| No augmentation | 0.944 | 0.981 |
| Class weights | 0.946 | 0.984 |
| Random oversampling | 0.939 | 0.981 |
| SMOTE | 0.948 | 0.985 |
| CTGAN | 0.948 | 0.980 |
| Ours (cWGAN-GP) | 0.949 | 0.981 |

### Per-class F1 on augmented classes (hgb)

| method | Bot | Web Attack |
|---|---|---|
| No augmentation | 0.839 | 0.944 |
| Class weights | 0.929 | 0.988 |
| Random oversampling | 0.948 | 0.981 |
| SMOTE | 0.945 | 0.982 |
| CTGAN | 0.936 | 0.957 |
| Ours (cWGAN-GP) | 0.931 | 0.979 |

![Per-class F1](figures/per_class_f1.png)

### How separable are the rare classes?

Share of each rare class's test rows whose exact feature vector also occurs in the training data under another label. No classifier can be sure of those rows, whatever augmentation is used.

| class | n_test | test_seen_with_other_label | test_seen_with_same_label_only | test_unseen | train_shared_with_other_label |
|---|---|---|---|---|---|
| Bot | 292 | 1.4 % | 27.7 % | 70.9 % | 1.2 % |
| Web Attack | 322 | 0.0 % | 0.0 % | 100.0 % | 0.0 % |

## Synthetic data quality (augmented classes, vs. real test rows)

C2ST AUC: 0.5 = indistinguishable from real. DCR ratio: ~1 = as novel as real held-out traffic, 0 = copies of training rows.

| method | c2st_auc | mmd | precision | recall | coverage | wasserstein | correlation_gap | dcr_ratio | exact_copy_rate |
|---|---|---|---|---|---|---|---|---|---|
| ros | 0.563 ± 0.014 | -0.001 ± 0.000 | 0.971 ± 0.003 | 0.874 ± 0.005 | 1.000 ± 0.000 | 0.011 ± 0.002 | 0.016 ± 0.001 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| smote | 0.585 ± 0.025 | -0.001 ± 0.000 | 0.977 ± 0.000 | 0.851 ± 0.012 | 1.000 ± 0.000 | 0.012 ± 0.002 | 0.016 ± 0.002 | 0.352 ± 0.010 | 0.124 ± 0.004 |
| ctgan | 1.000 ± 0.000 | 0.342 ± 0.022 | 0.000 ± 0.000 | 0.523 ± 0.325 | 0.001 ± 0.001 | 0.263 ± 0.016 | 0.233 ± 0.012 | 1454.176 ± 59.267 | 0.000 ± 0.000 |
| gan | 0.998 ± 0.001 | 0.185 ± 0.001 | 0.153 ± 0.006 | 0.907 ± 0.012 | 0.045 ± 0.002 | 0.027 ± 0.003 | 0.037 ± 0.003 | 63.741 ± 4.549 | 0.000 ± 0.000 |

![Fidelity vs privacy](figures/fidelity_vs_privacy.png)

### Do synthetic rows look like their own class?

Share of synthetic rows that a random forest trained on real data assigns to the intended class (mean over seeds). Rows assigned to another class teach the detector false positives. The last row is the same classifier's recall on real test rows of each class, for reference.

| method | Bot | Web Attack |
|---|---|---|
| ros | 1.00 | 1.00 |
| smote | 0.98 | 0.98 |
| ctgan | 0.00 | 0.00 |
| gan | 0.61 | 0.53 |
| *real test rows* | 0.96 | 0.98 |

### Conditional sampling

Share of CTGAN rows whose generated label matched the requested class; mismatches are rejected and drawn again.

| method | class | requested | generated | condition_hit_rate |
|---|---|---|---|---|
| ctgan | Bot | 8592.0 | 8592.0 | 0.0245 |
| ctgan | Web Attack | 8452.0 | 8452.0 | 0.0152 |

## Train on synthetic only (real rows of augmented classes removed, RF)

| method | rare_macro_f1 | rare_macro_recall | macro_f1 |
|---|---|---|---|
| ros | 0.961 ± 0.001 | 0.977 ± 0.003 | 0.992 ± 0.000 |
| smote | 0.966 ± 0.002 | 0.973 ± 0.004 | 0.993 ± 0.000 |
| ctgan | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.816 ± 0.000 |
| gan | 0.758 ± 0.016 | 0.637 ± 0.018 | 0.954 ± 0.003 |

## GAN training

![Training curves](figures/training.png)
