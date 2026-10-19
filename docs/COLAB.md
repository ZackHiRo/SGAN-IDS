# Running the experiments on Google Colab

Everything is written to Google Drive, so a disconnect loses nothing. Re-running the
same command picks up where it stopped: data splits, trained generators and generated
synthetic data (including CTGAN's) are reused. Only the classifiers are refitted.

Select a GPU runtime (Runtime → Change runtime type), then run these cells.

**1. Mount Drive and get the code**

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%%bash
cd /content
[ -d SGAN-IDS ] || git clone https://github.com/ZackHiRo/SGAN-IDS.git
cd SGAN-IDS && git pull
```

**2. Datasets (once)**

Put the raw CSVs in `MyDrive/sgan-ids/datasets/` with the layout from
[`scripts/download_data.md`](../scripts/download_data.md) (`unsw_nb15/…`, `cic_ids2017/…`).
Copy them to the local disk at the start of each session, since reading from local disk
is much faster than reading from Drive:

```bash
%%bash
mkdir -p /content/SGAN-IDS/datasets
cp -rn /content/drive/MyDrive/sgan-ids/datasets/* /content/SGAN-IDS/datasets/
```

**3. Run one dataset per session**

```bash
%%bash
cd /content/SGAN-IDS
export ARTIFACTS=/content/drive/MyDrive/sgan-ids/artifacts
export RESULTS=/content/drive/MyDrive/sgan-ids/results
bash scripts/run_gpu_pod.sh unsw_nb15      # then, in another session: cic_ids2017
```

Rough durations on a Colab GPU:

- **UNSW-NB15:** about 3–3.5 h. The first run measured 4 h 15 min, 2.5 h of it CTGAN, before CTGAN's training rows were capped.
- **CIC-IDS2017:** estimated at 5–6 h; not measured yet.

If the session ends early, run the same cell again.

**4. Bring the results back**

`MyDrive/sgan-ids/results/<dataset>/` holds `report.md`, the result CSVs, figures and the
run log. Download that folder (or zip it), unzip it into the repository's `results/`,
and commit it.
