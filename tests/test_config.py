import pytest

from sgan_ids.config import load_config


def test_overrides_are_parsed_as_yaml():
    cfg = load_config(
        "configs/toy.yaml", ["gan.n_steps=7", "seeds=[5, 6]", "data.label_map={a: b}"]
    )
    assert cfg.gan.n_steps == 7
    assert cfg.seeds == [5, 6]
    assert cfg.data.label_map == {"a": "b"}
    assert cfg.gan.betas == (0.5, 0.9)


def test_unknown_keys_fail_loudly():
    with pytest.raises(ValueError, match="Unknown GANConfig keys"):
        load_config("configs/toy.yaml", ["gan.n_step=7"])


@pytest.mark.parametrize(
    "path", ["configs/unsw_nb15.yaml", "configs/cic_ids2017.yaml", "configs/toy.yaml"]
)
def test_shipped_configs_load(path):
    load_config(path)
