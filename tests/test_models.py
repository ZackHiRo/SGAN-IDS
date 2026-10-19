import numpy as np
import torch

from sgan_ids.config import GANConfig
from sgan_ids.models import gradient_penalty
from sgan_ids.training.trainer import ClassBalancedSampler, build_models


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
