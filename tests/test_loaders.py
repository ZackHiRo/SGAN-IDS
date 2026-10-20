"""The real loaders, run on tiny CSVs written in the exact raw formats (no download needed)."""

import numpy as np
import pandas as pd
import pytest

from sgan_ids.config import DataConfig
from sgan_ids.data import prepare
from sgan_ids.data.loaders import _clean_label, load_cic_ids2017, load_unsw_nb15

UNSW_COLUMNS = (
    "id,dur,proto,service,state,spkts,dpkts,sbytes,dbytes,rate,sttl,dttl,sload,dload,sloss,dloss,"
    "sinpkt,dinpkt,sjit,djit,swin,stcpb,dtcpb,dwin,tcprtt,synack,ackdat,smean,dmean,trans_depth,"
    "response_body_len,ct_srv_src,ct_state_ttl,ct_dst_ltm,ct_src_dport_ltm,ct_dst_sport_ltm,"
    "ct_dst_src_ltm,is_ftp_login,ct_ftp_cmd,ct_flw_http_mthd,ct_src_ltm,ct_srv_dst,is_sm_ips_ports,"
    "attack_cat,label"
).split(",")


def _unsw(n: int, rng: np.random.Generator) -> pd.DataFrame:
    frame = pd.DataFrame(rng.integers(0, 1000, (n, len(UNSW_COLUMNS))), columns=UNSW_COLUMNS)
    frame["id"] = np.arange(1, n + 1)
    frame["proto"] = rng.choice(["tcp", "udp", "arp"], n)
    frame["service"] = rng.choice(["-", "http", "dns"], n)
    frame["state"] = rng.choice(["FIN", "INT", "CON"], n)
    frame["attack_cat"] = rng.choice(
        ["Normal", "Generic", "Exploits", "Worms"], n, p=[0.5, 0.3, 0.15, 0.05]
    )
    frame["label"] = (frame["attack_cat"] != "Normal").astype(int)
    return frame


def test_unsw_nb15_loader_and_official_split(tmp_path):
    rng = np.random.default_rng(0)
    folder = tmp_path / "unsw_nb15"
    folder.mkdir()
    _unsw(800, rng).to_csv(folder / "UNSW_NB15_training-set.csv", index=False)
    _unsw(300, rng).to_csv(folder / "UNSW_NB15_testing-set.csv", index=False)

    frame, categorical, is_test = load_unsw_nb15(tmp_path)
    assert categorical == ["proto", "service", "state"]
    assert "id" not in frame and "attack_cat" not in frame
    assert list(frame.columns).count("label") == 1  # binary label dropped, attack_cat renamed
    assert is_test.sum() == 300

    data = prepare(DataConfig(dataset="unsw_nb15", root=str(tmp_path)))
    assert data.report["split"] == "official"
    assert len(data.X_test) == 300  # official test partition untouched
    assert data.classes == ["Exploits", "Generic", "Normal", "Worms"]


def test_unsw_nb15_swapped_file_names_are_detected(tmp_path):
    """Some mirrors swap the two names; the larger file must still become the training set."""
    rng = np.random.default_rng(1)
    folder = tmp_path / "unsw_nb15"
    folder.mkdir()
    _unsw(300, rng).to_csv(folder / "UNSW_NB15_training-set.csv", index=False)
    _unsw(800, rng).to_csv(folder / "UNSW_NB15_testing-set.csv", index=False)

    _, _, is_test = load_unsw_nb15(tmp_path)
    assert is_test.sum() == 300
    data = prepare(DataConfig(dataset="unsw_nb15", root=str(tmp_path)))
    assert len(data.X_test) == 300


def test_cic_ids2017_loader_quirks(tmp_path):
    rng = np.random.default_rng(0)
    folder = tmp_path / "cic_ids2017"
    folder.mkdir()
    n = 400
    labels = rng.choice(
        [
            "BENIGN",
            "DoS Hulk",
            "Web Attack \u2013 XSS",
            "Web Attack \u2013 Brute Force",
            "Heartbleed",
        ],
        n,
        p=[0.6, 0.2, 0.08, 0.08, 0.04],
    )
    frame = pd.DataFrame(
        {
            " Destination Port": rng.choice([80, 443, 22], n),
            " Flow Duration": rng.integers(0, 10**7, n),
            "Flow Bytes/s": rng.exponential(1e5, n).astype(str),
            " Fwd Header Length": rng.integers(0, 100, n),
            " Fwd Header Length.1": rng.integers(0, 100, n),
            " Label": labels,
        }
    )
    frame.loc[:4, "Flow Bytes/s"] = "Infinity"
    frame.loc[5:7, "Flow Bytes/s"] = np.nan
    half = n // 2
    frame[:half].to_csv(folder / "Monday.csv", index=False, encoding="cp1252")
    frame[half:].to_csv(folder / "Thursday.csv", index=False, encoding="cp1252")

    raw, categorical, official = load_cic_ids2017(tmp_path)
    assert official is None and categorical == ["Destination Port"]
    assert len(raw) == n - 8  # Infinity and NaN rows removed
    assert "Fwd Header Length.1" not in raw
    assert "Web Attack - XSS" in set(raw["label"])

    cfg = DataConfig(
        dataset="cic_ids2017",
        root=str(tmp_path),
        drop_classes=["Heartbleed"],
        label_map={"Web Attack - XSS": "Web Attack", "Web Attack - Brute Force": "Web Attack"},
    )
    data = prepare(cfg)
    assert data.classes == ["BENIGN", "DoS Hulk", "Web Attack"]


@pytest.mark.parametrize(
    "raw",
    [
        "Web Attack \x96 XSS",  # original cp1252 byte read as latin-1
        "Web Attack \u2013 XSS",  # proper en dash
        "Web Attack \ufffd XSS",  # lossy UTF-8 conversion
        "Web Attack \u00ef\u00bf\u00bd XSS",  # that conversion read as latin-1 (seen on Colab)
        "  Web Attack  -  XSS ",
    ],
)
def test_web_attack_labels_normalise_to_one_form(raw):
    assert _clean_label(raw) == "Web Attack - XSS"
