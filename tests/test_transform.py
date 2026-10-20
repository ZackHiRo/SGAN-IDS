import numpy as np
import pandas as pd

from sgan_ids.data.loaders import LABEL, load_toy
from sgan_ids.data.transform import TabularTransformer


def _fit():
    frame, categorical, _ = load_toy()
    features = frame.drop(columns=LABEL)
    return features, TabularTransformer(categorical).fit(features)


def test_encoded_range_and_layout():
    features, tf = _fit()
    x = tf.transform(features)
    assert x.shape[1] == tf.n_features == tf.spans[-1].end
    assert x.min() >= -1 and x.max() <= 1
    assert "constant" in tf.dropped_constant
    assert "bytes" in [c for c, flag in zip(tf.numeric, tf.log) if flag]


def test_round_trip_is_exact_for_integers_and_categories():
    features, tf = _fit()
    back = tf.inverse_transform(tf.transform(features))
    np.testing.assert_array_equal(back["proto"].to_numpy(), features["proto"].to_numpy())
    np.testing.assert_array_equal(back["packets"].to_numpy(), features["packets"].to_numpy())
    np.testing.assert_allclose(
        back["duration"].to_numpy(), features["duration"].to_numpy(), rtol=1e-5, atol=1e-6
    )


def test_project_returns_valid_records():
    features, tf = _fit()
    noisy = tf.transform(features) + np.random.default_rng(0).normal(
        0, 0.3, (len(features), tf.n_features)
    )
    projected = tf.project(noisy)
    assert np.allclose(tf.project(projected), projected, atol=1e-6)  # idempotent
    for span in tf.spans:
        if span.kind == "categorical":
            assert np.all(projected[:, span.start : span.end].sum(axis=1) == 1)
    raw = tf.inverse_transform(projected)
    assert np.all(raw["packets"] == np.round(raw["packets"]))


def test_low_cardinality_numerics_become_categories():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "ttl": rng.choice([0, 62, 254], 500),  # few distinct values -> categorical
            "bytes": rng.lognormal(5, 2, 500),  # continuous
            "proto": rng.choice(["tcp", "udp"], 500),
        }
    )
    tf = TabularTransformer(["proto"], max_discrete_values=32).fit(frame)
    assert tf.discrete == ["ttl"] and tf.numeric == ["bytes"]
    x = tf.transform(frame)
    np.testing.assert_array_equal(tf.inverse_transform(x)["ttl"], frame["ttl"].astype(float))

    # Whatever the generator outputs, decoded TTLs are values that really occur.
    noisy = x + rng.normal(0, 0.5, x.shape)
    assert set(tf.inverse_transform(tf.project(noisy))["ttl"]) <= {0.0, 62.0, 254.0}


def test_discrete_encoding_can_be_disabled():
    frame = pd.DataFrame({"ttl": [0, 62, 254, 62] * 10, "proto": ["tcp", "udp"] * 20})
    tf = TabularTransformer(["proto"], max_discrete_values=0).fit(frame)
    assert tf.discrete == [] and tf.numeric == ["ttl"]


def test_frequent_exact_values_are_reproduced_exactly():
    rng = np.random.default_rng(0)
    n = 1000
    y = np.r_[np.zeros(900, int), np.ones(100, int)]
    jitter = np.where(rng.random(n) < 0.5, 0.0, rng.exponential(50.0, n))  # half exactly 0
    size = np.where(y == 1, 45.0, rng.integers(40, 1500, n)).astype(float)  # class 1 always 45
    frame = pd.DataFrame({"jitter": jitter, "size": size})
    tf = TabularTransformer([], mode_min_share=0.05, mode_min_rows=10).fit(frame, y)
    modes = dict(zip(tf.numeric, tf.modes))
    assert 0.0 in modes["jitter"]
    assert 45.0 in modes["size"]  # common in class 1 only, found because modes are per class

    x = tf.transform(frame)
    back = tf.inverse_transform(x)
    assert np.mean(back["jitter"] == 0.0) == np.mean(jitter == 0.0)
    np.testing.assert_allclose(back["size"], size)

    # A generator output that picks the "0" mode decodes to exactly 0, not 0.0003.
    noisy = tf.project(x + rng.normal(0, 0.05, x.shape))
    assert np.mean(tf.inverse_transform(noisy)["jitter"] == 0.0) > 0.3
