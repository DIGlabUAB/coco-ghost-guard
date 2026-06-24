from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Union

import requests


PathLike = Union[str, Path]


def encode_image(path: PathLike) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_ollama(
    image_path: PathLike,
    prompt: str,
    model: str,
    host: str,
    temperature: float = 0,
    num_ctx: int = 2048,
    timeout: int = 180,
) -> tuple[str, dict[str, Any]]:
    img_b64 = encode_image(image_path)
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }
    response = requests.post(f"{host.rstrip('/')}/api/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("response", ""), data


def list_ollama_models(host: str, timeout: int = 10) -> list[str]:
    response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return [model.get("name", "") for model in data.get("models", []) if model.get("name")]
