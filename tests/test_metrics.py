import numpy as np
import pytest

from sgan_ids.augment import augmentation_plan, resolve_classes
from sgan_ids.evaluation.fidelity import c2st_auc, median_gamma, mmd_rbf, prdc
from sgan_ids.evaluation.privacy import dcr_ratio

rng = np.random.default_rng(0)
A = rng.normal(size=(400, 5))
B = rng.normal(size=(400, 5))
SHIFTED = rng.normal(loc=2.0, size=(400, 5))


def test_mmd_separates_distributions():
    gamma = median_gamma(A)
    assert abs(mmd_rbf(A, B, gamma)) < 0.01 < mmd_rbf(A, SHIFTED, gamma)


def test_prdc_on_identical_and_disjoint_sets():
    same = prdc(A, A.copy())
    assert same["precision"] == 1 and same["coverage"] == 1
    far = prdc(A, A + 100)
    assert far["precision"] == 0 and far["coverage"] == 0


def test_c2st_is_chance_level_for_same_distribution():
    assert 0.4 < c2st_auc(A, B) < 0.6
    assert c2st_auc(A, SHIFTED) > 0.95


def test_dcr_flags_copies():
    assert dcr_ratio(A, B, A[:100])["exact_copy_rate"] == 1
    assert 0.7 < dcr_ratio(A, B, rng.normal(size=(400, 5)))["dcr_ratio"] < 1.3


def test_plan_only_tops_up_small_classes():
    y = np.r_[np.zeros(50), np.ones(5), np.full(20, 2)].astype(int)
    assert augmentation_plan(y, 4, 30) == {1: 25, 2: 10}


def test_plan_restricted_to_configured_classes():
    y = np.r_[np.zeros(50), np.ones(5), np.full(20, 2)].astype(int)
    assert augmentation_plan(y, 3, 30, classes=[1]) == {1: 25}
    assert augmentation_plan(y, 3, 30, classes=[0]) == {}  # already above target


def test_resolve_classes_rejects_unknown_names():
    assert resolve_classes([], ["a", "b"]) is None
    assert resolve_classes(["b"], ["a", "b"]) == [1]
    with pytest.raises(ValueError, match="not found"):
        resolve_classes(["Worm"], ["a", "b"])


def test_label_overlap_counts_conflicting_rows():
    from sgan_ids.evaluation.overlap import label_overlap

    X_train = np.array([[0, 0], [0, 0], [1, 1], [2, 2]], dtype=np.float32)
    y_train = np.array([0, 1, 1, 1])  # row [0,0] appears as class 0 and class 1
    X_test = np.array([[0, 0], [1, 1], [3, 3], [2, 2]], dtype=np.float32)
    y_test = np.array([1, 1, 1, 0])
    (row,) = label_overlap(X_train, y_train, X_test, y_test, [1])
    assert row["n_test"] == 3
    assert row["test_seen_with_other_label"] == 1 / 3  # [0,0]
    assert row["test_seen_with_same_label_only"] == 1 / 3  # [1,1]
    assert row["test_unseen"] == pytest.approx(1 / 3)  # [3,3]
    assert row["train_shared_with_other_label"] == 1 / 3
