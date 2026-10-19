# Getting the datasets

Raw data goes under `datasets/` (git-ignored). Expected layout:

```text
datasets/
├── unsw_nb15/
│   ├── UNSW_NB15_training-set.csv   # 175,341 rows
│   └── UNSW_NB15_testing-set.csv    #  82,332 rows
└── cic_ids2017/
    ├── Monday-WorkingHours.pcap_ISCX.csv
    ├── ...                          # the 8 MachineLearningCVE CSVs
    └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

## UNSW-NB15 (≈ 45 MB)

Official page: <https://research.unsw.edu.au/projects/unsw-nb15-dataset>. Download the
"training and testing set" partitions (`UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`).
Community Kaggle mirror (check that the two partition files are inside):

```bash
kaggle datasets download -d mrwellsdavid/unsw-nb15 -p datasets/unsw_nb15 --unzip
```

Only the two partition files are used; the four raw `UNSW-NB15_[1-4].csv` files can be deleted.

## CIC-IDS2017 (≈ 850 MB uncompressed)

Official page: <https://www.unb.ca/cic/datasets/ids-2017.html> → `MachineLearningCSV.zip`.
Unzip and move the 8 CSVs from `MachineLearningCVE/` into `datasets/cic_ids2017/`.

```bash
unzip MachineLearningCSV.zip && mv MachineLearningCVE/*.csv datasets/cic_ids2017/
```

Use the `MachineLearningCVE` version, not `GeneratedLabelledFlows`: the latter contains IPs and
timestamps, which the loader would drop anyway.
