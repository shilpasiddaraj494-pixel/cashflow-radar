"""Database access. PostgreSQL in Docker, SQLite file for local runs and tests."""
from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from .config import resolve


def database_url() -> str:
    return os.environ.get("DATABASE_URL", f"sqlite:///{resolve('artifacts/cashflow_radar.db').as_posix()}")


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    url = url or database_url()
    if url.startswith("sqlite:///"):
        resolve(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, pool_pre_ping=True, future=True)
