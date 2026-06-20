"""Backend selection and the narrow pyclesperanto adapter contract."""

from __future__ import annotations

from typing import Any

_backend: Any | None = None


def get_backend() -> Any:
    global _backend
    if _backend is None:
        import pyclesperanto as cle

        _backend = cle
    return _backend


def set_backend(backend: Any) -> None:
    """Set the backend (primarily useful for tests and downstream adapters)."""
    global _backend
    _backend = backend
