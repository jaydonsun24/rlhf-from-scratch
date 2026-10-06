from dataclasses import dataclass

import pytest

from rlhf.utils.config import load_config, load_model_config


@dataclass
class DemoConfig:
    model: str = "tiny"
    lr: float = 3e-4
    max_steps: int = 100
    compile: bool = True


def test_yaml_then_cli_overrides(tmp_path):
    path = tmp_path / "demo.yaml"
    path.write_text("lr: 6e-4\nmax_steps: 10\n")  # PyYAML reads 6e-4 as a *string*
    cfg = load_config(DemoConfig, ["--config", str(path), "max_steps=50", "compile=false"])
    assert cfg.lr == 6e-4 and isinstance(cfg.lr, float)
    assert cfg.max_steps == 50  # CLI beats YAML
    assert cfg.compile is False
    assert cfg.model == "tiny"  # untouched default


def test_scientific_int():
    assert load_config(DemoConfig, ["max_steps=5e3"]).max_steps == 5000


def test_bad_values_fail_loudly():
    with pytest.raises(KeyError):
        load_config(DemoConfig, ["learning_rate=1e-3"])  # typo'd key
    with pytest.raises(ValueError):
        load_config(DemoConfig, ["max_steps=1.5"])
    with pytest.raises(ValueError):
        load_config(DemoConfig, ["compile=maybe"])


def test_model_presets():
    tiny = load_model_config("tiny")
    assert (tiny.n_layer, tiny.d_model, tiny.context_len) == (2, 128, 256)
    assert tiny.vocab_size == 50304  # not in the YAML, so the dataclass default applies
    big = load_model_config("gpt124m")
    assert (big.n_layer, big.n_head, big.d_model, big.d_ff) == (12, 12, 768, 2048)
