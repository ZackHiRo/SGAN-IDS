import numpy as np
import torch

from sgan_ids.config import GANConfig
from sgan_ids.models import gradient_penalty
from sgan_ids.training.trainer import (
    MIN_VAL_ROWS,
    ClassBalancedSampler,
    ValidationScorer,
    build_models,
)


def test_generator_respects_feature_types(toy_data):
    G, D = build_models(
        GANConfig(), toy_data.X_train.shape[1], toy_data.n_classes, toy_data.transformer.spans
    )
    y = torch.randint(0, toy_data.n_classes, (64,))
    x = G.sample(y)
    for span in toy_data.transformer.spans:
        block = x[:, span.start : span.end]
        if span.kind == "numeric":
            assert block.abs().max() <= 1
        else:
            assert torch.all(block.sum(1) == 1) and torch.all((block == 0) | (block == 1))
    assert D(x, y).shape == (64,)


def test_projection_critic_depends_on_label(toy_data):
    _, D = build_models(
        GANConfig(), toy_data.X_train.shape[1], toy_data.n_classes, toy_data.transformer.spans
    )
    torch.nn.init.normal_(D.embed.weight, std=1.0)
    x = torch.as_tensor(toy_data.X_train[:8])
    assert not torch.allclose(
        D(x, torch.zeros(8, dtype=torch.long)), D(x, torch.ones(8, dtype=torch.long))
    )


def test_gradient_penalty_is_finite_and_differentiable(toy_data):
    G, D = build_models(
        GANConfig(), toy_data.X_train.shape[1], toy_data.n_classes, toy_data.transformer.spans
    )
    real = torch.as_tensor(toy_data.X_train[:32])
    y = torch.as_tensor(toy_data.y_train[:32])
    gp = gradient_penalty(D, real, G.sample(y), y)
    gp.backward()
    assert torch.isfinite(gp) and any(p.grad is not None for p in D.parameters())


def test_sampler_follows_alpha():
    y = np.r_[np.zeros(9000, int), np.ones(1000, int)]
    for alpha, expected in [(1.0, 0.9), (0.0, 0.5), (0.5, 3 / 4)]:
        s = ClassBalancedSampler(y, 2, alpha, torch.device("cpu"))
        labels = s.labels(20000)
        assert abs((labels == 0).float().mean().item() - expected) < 0.02
        assert torch.all(torch.as_tensor(y)[s.rows(labels)] == labels)


def test_validation_score_is_deterministic_and_leaves_training_rng_alone(toy_data):
    G, _ = build_models(
        GANConfig(), toy_data.X_train.shape[1], toy_data.n_classes, toy_data.transformer.spans
    )
    scorer = ValidationScorer(toy_data.X_val, toy_data.y_val, toy_data.n_classes, 100, seed=0)
    torch.manual_seed(123)
    expected_next = torch.rand(1)
    torch.manual_seed(123)
    first = scorer(G, torch.device("cpu"))
    assert torch.equal(torch.rand(1), expected_next)  # RNG stream unchanged by scoring
    assert scorer(G, torch.device("cpu")) == first


def test_validation_score_skips_tiny_classes():
    y = np.r_[np.zeros(50, int), np.ones(MIN_VAL_ROWS - 1, int)]
    X = np.random.default_rng(0).normal(size=(len(y), 3)).astype(np.float32)
    scorer = ValidationScorer(X, y, 2, 30, seed=0)
    assert list(scorer.reference) == [0] and scorer.excluded == [1]


def test_validation_bandwidth_is_per_class():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.01, (40, 3)), rng.normal(5, 1.0, (40, 3))]).astype(np.float32)
    y = np.r_[np.zeros(40, int), np.ones(40, int)]
    scorer = ValidationScorer(X, y, 2, 40, seed=0)
    # The tight class gets a much narrower kernel (larger gamma) than the spread-out one.
    assert scorer.gamma[0] > 100 * scorer.gamma[1]


def test_gan_samples_come_out_in_evaluation_space(toy_data):
    from sgan_ids.augment.methods import gan_sample

    G_train, _ = toy_data.generation_splits()
    G, _ = build_models(
        GANConfig(), G_train.shape[1], toy_data.n_classes, toy_data.gen_transformer.spans
    )
    X, y = gan_sample(G, toy_data, {2: 50, 3: 30}, seed=0)
    assert X.shape == (80, toy_data.X_train.shape[1])
    assert X.min() >= -1 and X.max() <= 1
    for span in toy_data.transformer.spans:
        if span.kind == "categorical":
            assert np.all(X[:, span.start : span.end].sum(axis=1) == 1)


def test_per_class_ctgan_fits_one_model_per_class(toy_data, monkeypatch):
    """Per-class CTGAN: each model sees only its class's rows; small classes get more epochs."""
    import sys
    import types

    from sgan_ids.augment.methods import ctgan_sample

    fitted = []

    class FakeCTGAN:
        def __init__(self, epochs, batch_size, verbose, enable_gpu):
            self.epochs, self.batch_size = epochs, batch_size

        def set_random_state(self, seed):
            self.seed = seed

        def fit(self, frame, discrete_columns):
            assert "label" not in frame.columns
            self.frame = frame
            fitted.append(self)

        def sample(self, n):
            return self.frame.sample(n, replace=True, random_state=self.seed)

    monkeypatch.setitem(sys.modules, "ctgan", types.SimpleNamespace(CTGAN=FakeCTGAN))
    plan = {2: 300, 3: 500}  # Recon (326 train rows), Worm (82)
    X, y, info = ctgan_sample(toy_data, plan, seed=0, epochs=10, batch_size=500, min_steps=100)

    counts = np.bincount(toy_data.y_train)
    assert [len(m.frame) for m in fitted] == [counts[2], counts[3]]
    assert all(m.batch_size % 10 == 0 and m.batch_size <= len(m.frame) for m in fitted)
    assert all(m.epochs * (len(m.frame) // m.batch_size) >= 100 for m in fitted)
    assert X.shape == (800, toy_data.X_train.shape[1]) and list(np.bincount(y)[2:]) == [300, 500]
    assert [r["class"] for r in info] == ["Recon", "Worm"]
