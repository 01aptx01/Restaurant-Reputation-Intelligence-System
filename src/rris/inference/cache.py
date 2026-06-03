# -*- coding: utf-8 -*-
"""In-process cache for heavy inference models (load once per process)."""

from __future__ import annotations

from typing import Any

_CACHE: dict[str, Any] = {}


def get(key: str) -> Any | None:
    return _CACHE.get(key)


def set(key: str, value: Any) -> Any:
    _CACHE[key] = value
    return value


def clear() -> None:
    _CACHE.clear()
