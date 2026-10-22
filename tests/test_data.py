import numpy as np


def test_splits_are_disjoint(toy_data):
    train = {r.tobytes() for r in toy_data.X_train}
    assert not any(r.tobytes() in train for r in toy_data.X_test)


def test_stratified_split_keeps_every_class(toy_data):
    for y in (toy_data.y_train, toy_data.y_val, toy_data.y_test):
        assert set(np.unique(y)) == set(range(toy_data.n_classes))


def test_report_records_preprocessing(toy_data):
    report = toy_data.report
    assert report["rows"]["train"] == len(toy_data.X_train)
    assert report["class_counts"]["test"]["Worm"] > 0


def test_evaluation_space_ignores_generation_settings():
    """Classifiers and baselines must see the same columns whatever the GAN's encoding is."""
    from sgan_ids.config import DataConfig
    from sgan_ids.data import prepare

    rich = prepare(DataConfig(dataset="toy", test_size=0.2), seed=0)
    plain = prepare(
        DataConfig(dataset="toy", test_size=0.2, max_discrete_values=0, max_modes_per_column=0),
        seed=0,
    )
    np.testing.assert_array_equal(rich.X_train, plain.X_train)
    np.testing.assert_array_equal(rich.X_test, plain.X_test)
    assert rich.G_train.shape[1] > rich.X_train.shape[1]  # discrete one-hots and mode selectors


def test_evaluation_space_has_one_column_per_numeric_feature(toy_data):
    tf = toy_data.transformer
    assert tf.discrete == [] and all(len(m) == 0 for m in tf.modes)
    numeric = [s for s in tf.spans if s.kind == "numeric"]
    assert [s.name for s in numeric] == tf.numeric
    assert {s.name for s in tf.spans if s.kind == "categorical"} == {"proto"}


def test_generation_splits_are_saved_and_loaded_lazily(toy_data, tmp_path):
    from sgan_ids.data import PreparedData

    fingerprint = toy_data.fingerprint()
    toy_data.save(tmp_path)
    loaded = PreparedData.load(tmp_path)
    assert loaded.G_train is None
    G_train, _ = loaded.generation_splits()
    np.testing.assert_array_equal(G_train, toy_data.G_train)
    assert loaded.fingerprint() == fingerprint
    loaded.release_generation_splits()
    assert loaded.G_train is None
