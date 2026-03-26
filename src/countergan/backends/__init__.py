"""Backend registry and auto-detection.

Lazily imports framework backends so that ``import countergan`` never
pulls in torch or tensorflow at the top level.
"""

from __future__ import annotations

import importlib

from countergan._backend import Backend

_BACKEND_NAMES = {"torch", "tensorflow"}


def detect_backend(preference: str | None = None) -> str:
    """Return the name of the backend to use.

    Priority
    --------
    1. Explicit *preference* (if provided and importable).
    2. ``torch`` (if installed).
    3. ``tensorflow`` (if installed).
    4. Raise :class:`ImportError`.
    """
    if preference is not None:
        if preference not in _BACKEND_NAMES:
            raise ValueError(
                f"Unknown backend {preference!r}. Choose from {sorted(_BACKEND_NAMES)}"
            )
        _check_importable(preference)
        return preference

    for name in ("torch", "tensorflow"):
        try:
            _check_importable(name)
            return name
        except ImportError:
            continue

    raise ImportError(
        "No supported backend found. Install one of:\n"
        "  pip install countergan[torch]\n"
        "  pip install countergan[tensorflow]"
    )


def get_backend(name: str) -> Backend:
    """Lazily import and instantiate the named backend."""
    if name == "torch":
        from countergan.backends._torch import TorchBackend

        return TorchBackend()
    if name == "tensorflow":
        from countergan.backends._tensorflow import TensorFlowBackend

        return TensorFlowBackend()
    raise ValueError(f"Unknown backend: {name!r}")


def _check_importable(name: str) -> None:
    """Raise :class:`ImportError` if the framework package is not installed."""
    try:
        importlib.import_module(name)
    except ImportError:
        raise ImportError(
            f"Backend {name!r} requested but package is not installed. "
            f"Install it with: pip install countergan[{name}]"
        ) from None
