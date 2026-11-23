"""Load the YAML config once and expose it as a plain dict."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# CFR_HOME lets an installed (non-editable) package find config/, data/ and artifacts/.
PROJECT_ROOT = Path(os.environ.get("CFR_HOME", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path or os.environ.get("CFR_CONFIG", DEFAULT_CONFIG))
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve(path: str) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
