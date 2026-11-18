"""Locate OpenCV Haar cascade XML files.

OpenCV 5.x wheels no longer ship the XML files, and OpenCV on Windows cannot
open paths containing non-ASCII characters (e.g. a Cyrillic workspace path).
So we bundle the cascades in ``clipper_bot/cascades/`` and copy them once into
an ASCII-only temp directory before handing them to cv2.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

LOCAL_DIR = Path(__file__).resolve().parent / "cascades"
_CACHE_DIR = Path(tempfile.gettempdir()) / "clipper_cascades"


def _ascii_copy(source: Path) -> Path:
    """Return an ASCII-only path containing a copy of *source*."""
    try:
        str(source).encode("ascii")
        return source
    except UnicodeEncodeError:
        pass

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _CACHE_DIR / source.name
    if not cached.is_file() or cached.stat().st_size != source.stat().st_size:
        shutil.copy2(source, cached)
    return cached


def find(name: str) -> Optional[Path]:
    local = LOCAL_DIR / name
    if local.is_file():
        return _ascii_copy(local)

    try:
        import cv2

        builtin = Path(cv2.data.haarcascades) / name
        if builtin.is_file():
            return _ascii_copy(builtin)
    except Exception:  # noqa: BLE001
        pass
    return None
