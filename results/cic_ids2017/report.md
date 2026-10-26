# Results: cic_ids2017

Dataset `cic_ids2017`, split: stratified test_size=0.15. Rows: {'train': 488188, 'val': 86151, 'test': 101355}. Encoded features: 77 (GAN generation space: 694).

Augmented classes (rows added): {'Bot': 8592, 'Web Attack': 8452}

Seeds: [0, 1, 2]. Values are mean ± std over seeds; all scores are on the real, untouched test split.

## Detection (train on real + synthetic, test on real)

| classifier | method | macro_f1 | rare_macro_f1 | rare_macro_recall | balanced_accuracy |
|---|---|---|---|---|---|
| rf | none | 0.992 ± 0.000 | 0.963 ± 0.002 | 0.968 ± 0.002 | 0.993 ± 0.000 |
| rf | class_weight | 0.992 ± 0.000 | 0.964 ± 0.002 | 0.971 ± 0.003 | 0.993 ± 0.000 |
| rf | balanced_weight | 0.993 ± 0.000 | 0.965 ± 0.002 | 0.975 ± 0.002 | 0.994 ± 0.001 |
| rf | ros | 0.991 ± 0.001 | 0.960 ± 0.003 | 0.978 ± 0.005 | 0.994 ± 0.001 |
| rf | smote | 0.993 ± 0.000 | 0.966 ± 0.002 | 0.984 ± 0.001 | 0.996 ± 0.000 |
| rf | ctgan | 0.992 ± 0.000 | 0.965 ± 0.000 | 0.974 ± 0.002 | 0.994 ± 0.000 |
| rf | gan | 0.993 ± 0.000 | 0.966 ± 0.001 | 0.977 ± 0.003 | 0.994 ± 0.000 |
| hgb | none | 0.970 ± 0.006 | 0.891 ± 0.024 | 0.909 ± 0.057 | 0.975 ± 0.012 |
| hgb | class_weight | 0.980 ± 0.005 | 0.956 ± 0.001 | 0.983 ± 0.002 | 0.988 ± 0.004 |
| hgb | balanced_weight | 0.991 ± 0.000 | 0.959 ± 0.001 | 0.993 ± 0.001 | 0.998 ± 0.000 |
| hgb | ros | 0.987 ± 0.002 | 0.964 ± 0.003 | 0.984 ± 0.001 | 0.992 ± 0.002 |
| hgb | smote | 0.983 ± 0.007 | 0.963 ± 0.003 | 0.987 ± 0.001 | 0.991 ± 0.002 |
| hgb | ctgan | 0.984 ± 0.004 | 0.962 ± 0.006 | 0.983 ± 0.008 | 0.989 ± 0.006 |
| hgb | gan | 0.983 ± 0.004 | 0.952 ± 0.004 | 0.979 ± 0.006 | 0.989 ± 0.004 |

### Rare-class F1 change vs. no augmentation (paired by seed and classifier)

| method | mean Δ rare-F1 | seeds×classifiers improved |
|---|---|---|
| class_weight | +0.033 | 5/6 |
| balanced_weight | +0.035 | 6/6 |
| ros | +0.035 | 3/6 |
| smote | +0.037 | 6/6 |
| ctgan | +0.036 | 6/6 |
| gan | +0.032 | 6/6 |

### Per-class F1 on augmented classes (rf)

| method | Bot | Web Attack |
|---|---|---|
| No augmentation | 0.944 | 0.981 |
| Class weights (rare classes) | 0.946 | 0.982 |
| Balanced class weights | 0.946 | 0.984 |
| Random oversampling | 0.939 | 0.981 |
| SMOTE | 0.948 | 0.985 |
| CTGAN | 0.946 | 0.983 |
| Ours (cWGAN-GP) | 0.951 | 0.981 |

### Per-class F1 on augmented classes (hgb)

| method | Bot | Web Attack |
|---|---|---|
| No augmentation | 0.839 | 0.944 |
| Class weights (rare classes) | 0.930 | 0.982 |
| Balanced class weights | 0.929 | 0.988 |
| Random oversampling | 0.948 | 0.981 |
| SMOTE | 0.943 | 0.982 |
| CTGAN | 0.945 | 0.980 |
| Ours (cWGAN-GP) | 0.929 | 0.974 |

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
| ctgan | 1.000 ± 0.000 | 0.285 ± 0.012 | 0.000 ± 0.000 | 0.825 ± 0.250 | 0.000 ± 0.000 | 0.144 ± 0.005 | 0.174 ± 0.003 | 1023.976 ± 17.698 | 0.000 ± 0.000 |
| gan | 0.998 ± 0.000 | 0.185 ± 0.002 | 0.149 ± 0.008 | 0.913 ± 0.017 | 0.048 ± 0.003 | 0.027 ± 0.003 | 0.038 ± 0.002 | 63.937 ± 3.507 | 0.000 ± 0.000 |

![Fidelity vs privacy](figures/fidelity_vs_privacy.png)

### Do synthetic rows look like their own class?

Share of synthetic rows that a random forest trained on real training data assigns to the intended class (mean over seeds). Compare each method with the last row, the same classifier's recall on real *held-out* rows: that is what genuinely new traffic of the class scores. Oversampling and SMOTE score high partly because their rows are copies of, or interpolations between, that classifier's own training rows; it does not mean they are better samples.

| method | Bot | Web Attack |
|---|---|---|
| ros | 1.00 | 1.00 |
| smote | 0.98 | 0.98 |
| ctgan | 0.03 | 0.00 |
| gan | 0.60 | 0.54 |
| *real test rows* | 0.96 | 0.98 |

### Generator details

Rows requested and generated per class. Conditional CTGAN: share of draws whose label matched the requested class (mismatches are rejected). Per-class CTGAN: training rows and epochs of each class's model.

| method | class | requested | generated | fit_rows | epochs |
|---|---|---|---|---|---|
| ctgan | Bot | 8592.0 | 8592.0 | 1408.0 | 1500.0 |
| ctgan | Web Attack | 8452.0 | 8452.0 | 1548.0 | 1000.0 |

## Train on synthetic only (real rows of augmented classes removed, RF)

| method | rare_macro_f1 | rare_macro_recall | macro_f1 |
|---|---|---|---|
| ros | 0.961 ± 0.001 | 0.977 ± 0.003 | 0.992 ± 0.000 |
| smote | 0.966 ± 0.002 | 0.973 ± 0.004 | 0.993 ± 0.000 |
| ctgan | 0.317 ± 0.002 | 0.213 ± 0.001 | 0.873 ± 0.000 |
| gan | 0.753 ± 0.005 | 0.633 ± 0.007 | 0.953 ± 0.001 |

## GAN training

![Training curves](figures/training.png)
