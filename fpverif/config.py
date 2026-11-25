from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config.yaml", overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        _set_path(cfg, key.split("."), yaml.safe_load(raw))
    return cfg


def _set_path(node: dict, keys: list[str], value: Any) -> None:
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config key, e.g. --set train.epochs=5")


def dataset_cfg(cfg: dict, name: str) -> dict:
    try:
        return cfg["datasets"][name]
    except KeyError as e:
        raise SystemExit(f"unknown dataset {name!r}; options: {list(cfg['datasets'])}") from e
