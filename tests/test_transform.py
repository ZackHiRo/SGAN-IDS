import numpy as np

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
