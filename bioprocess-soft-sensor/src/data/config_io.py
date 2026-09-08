"""Backward-compatible config helpers. Prefer ``src.config``."""

from __future__ import annotations

from typing import Any

from src.config import load_default, load_yaml


def load_config(name: str = "default.yaml") -> dict[str, Any]:
    if name in ("default.yaml", "default", ""):
        return load_default()
    return load_yaml(name)
