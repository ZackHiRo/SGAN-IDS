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
