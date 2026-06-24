from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Union

import yaml


ROOT = Path(__file__).resolve().parents[1]


def project_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


PathLike = Union[str, Path]


def load_config(path: PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config.")


def ensure_dirs(*paths: PathLike) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: PathLike, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def append_jsonl(path: PathLike, rows: list[dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: PathLike) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
