from __future__ import annotations

import argparse
import importlib
import sys

import requests

from .common import add_config_arg, load_config
from .ollama_client import list_ollama_models


REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "PIL",
    "matplotlib",
    "requests",
    "tqdm",
    "yaml",
    "pycocotools",
    "skimage",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local COCO-Ghost runtime setup.")
    add_config_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ollama = cfg["ollama"]

    if sys.version_info < (3, 9):
        raise SystemExit("[ERROR] Python 3.9+ is required")
    print("[OK] Python version")

    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise SystemExit(f"[ERROR] Missing packages: {', '.join(missing)}")
    print("[OK] Required packages")

    try:
        models = list_ollama_models(ollama["host"])
    except requests.RequestException as exc:
        raise SystemExit(f"[ERROR] Ollama server not reachable: {exc}") from exc
    print("[OK] Ollama server reachable")

    model = ollama["model"]
    if model not in models:
        available = ", ".join(models) if models else "(none)"
        raise SystemExit(f"[ERROR] Model not found: {model}. Available: {available}")
    print(f"[OK] Model found: {model}")


if __name__ == "__main__":
    main()
