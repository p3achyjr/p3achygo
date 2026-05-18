"""
Single source of truth for backend detection.

Precedence:
  1. If both P3ACHYGO_BACKEND and KERAS_BACKEND are set and disagree, raise.
  2. P3ACHYGO_BACKEND takes precedence over KERAS_BACKEND.
  3. Falls back to "torch" if neither is set.
"""

import os


def _norm(b):
    if b is None:
        return None
    return "tensorflow" if b in ("tf", "tensorflow") else b


_p3 = os.environ.get("P3ACHYGO_BACKEND")
_kb = os.environ.get("KERAS_BACKEND")
_p3n, _kbn = _norm(_p3), _norm(_kb)

if _p3n and _kbn and _p3n != _kbn:
    raise RuntimeError(
        f"P3ACHYGO_BACKEND={_p3!r} disagrees with KERAS_BACKEND={_kb!r}. "
        "Set them to the same value (or unset one) before importing backend_shim."
    )

BACKEND: str = _p3n or _kbn or "torch"
