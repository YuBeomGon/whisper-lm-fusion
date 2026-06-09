"""Backend factory and registry.

Select a backend by name in ``load(..., backend="ct2")``. Backends are referred
to by ``"module:Class"`` and imported only when actually created, so adding a
heavyweight backend never slows down ``import stt_wrapper``.

Register your own::

    from stt_wrapper.backends import register_backend
    register_backend("trtllm", "my_pkg.trtllm:TrtLlmBackend")
"""

from __future__ import annotations

import importlib

from stt_wrapper.backends.base import Backend, RawResult
from stt_wrapper.config import LoadConfig

_REGISTRY: dict[str, str] = {
    "ct2": "stt_wrapper.backends.ct2:Ct2Backend",
    # future: "trtllm", "hf-whisper", "openai-whisper"
}


def register_backend(name: str, target: str) -> None:
    """Register a backend under ``name`` as a ``"module:Class"`` target string."""
    _REGISTRY[name] = target


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def create_backend(name: str, config: LoadConfig) -> Backend:
    """Construct the backend registered as ``name`` from ``config``."""
    try:
        target = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown backend {name!r}; available: {available_backends()}"
        ) from None
    module_path, class_name = target.split(":")
    module = importlib.import_module(module_path)
    backend_cls = getattr(module, class_name)
    return backend_cls(config)


__all__ = [
    "Backend",
    "RawResult",
    "create_backend",
    "register_backend",
    "available_backends",
]
