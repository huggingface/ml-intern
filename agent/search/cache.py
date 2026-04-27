"""Small JSON cache for network-backed research search indexes."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60


def cache_root() -> Path:
    configured = os.environ.get("ML_INTERN_SEARCH_CACHE_DIR")
    if configured:
        return Path(configured)
    return Path.cwd() / ".ml-intern-cache" / "search"


def read_json(namespace: str, key: str, *, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> Any | None:
    path = _path(namespace, key)
    try:
        if time.time() - path.stat().st_mtime > ttl_seconds:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def write_json(namespace: str, key: str, value: Any) -> None:
    path = _path(namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value), encoding="utf-8")
    tmp.replace(path)


def stable_key(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _path(namespace: str, key: str) -> Path:
    safe_namespace = namespace.replace("/", "_")
    return cache_root() / safe_namespace / f"{key}.json"
