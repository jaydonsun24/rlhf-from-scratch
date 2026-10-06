"""Load a dataclass config from YAML, then apply `key=value` overrides from the command line.

Each phase defines its own config dataclass (e.g. PretrainConfig in train/pretrain.py)
with a `model: str` field naming a preset in configs/model/. Usage:

    cfg = load_config(PretrainConfig)          # reads --config and key=value from sys.argv
    model_cfg = load_model_config(cfg.model)   # e.g. "tiny" -> configs/model/tiny.yaml
"""

import argparse
import dataclasses
import sys
import typing
from pathlib import Path

import yaml

from rlhf.model.config import ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[3]  # src/rlhf/utils/config.py -> repo root
CONFIG_DIR = REPO_ROOT / "configs"


def _coerce(value, typ, key: str):
    """Convert a YAML or CLI value to the dataclass field's declared type."""
    if typ is bool:
        if isinstance(value, bool):
            return value
        text = str(value).lower()
        if text in ("true", "1", "yes"):
            return True
        if text in ("false", "0", "no"):
            return False
        raise ValueError(f"{key}: expected a bool, got {value!r}")
    if typ is int:
        # float() first so "5e9" works; then make sure it really is a whole number.
        number = float(value)
        if not number.is_integer():
            raise ValueError(f"{key}: expected an int, got {value!r}")
        return int(number)
    if typ is float:
        # PyYAML reads "6e-4" (no decimal point) as a string, so always convert.
        return float(value)
    if typ is str:
        return str(value)
    raise TypeError(f"{key}: unsupported config type {typ}")


def build_config(cls, values: dict):
    """Create `cls(**values)`, type-checking every key against the dataclass fields."""
    hints = typing.get_type_hints(cls)
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - field_names
    if unknown:
        raise KeyError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: _coerce(v, hints[k], k) for k, v in values.items()})


def parse_overrides(items: list[str]) -> dict:
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = value
    return overrides


def load_config(cls, argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="path to a YAML config")
    args, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    values = {}
    if args.config is not None:
        with open(args.config) as f:
            values = yaml.safe_load(f) or {}
    values.update(parse_overrides(rest))
    return build_config(cls, values)


def load_model_config(name: str) -> ModelConfig:
    path = CONFIG_DIR / "model" / f"{name}.yaml"
    with open(path) as f:
        values = yaml.safe_load(f) or {}
    return build_config(ModelConfig, values)


def save_config(cfg, path: Path) -> None:
    """Write the fully resolved config (after overrides) next to the run's outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)
