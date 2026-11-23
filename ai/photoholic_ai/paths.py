"""Where Photoholic's files live, both when run from source and inside the packaged app.

PyInstaller unpacks bundled files under ``sys._MEIPASS`` (the app's ``_internal`` folder):
``engine/image-reader.exe``, ``models/aesthetic_v1.onnx`` (+ ``.json``), ``app/static/``.
From source the same files are found in the repository.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

ENGINE_EXE = "image-reader.exe" if os.name == "nt" else "image-reader"
MODEL_NAME = "aesthetic_v1.onnx"


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_dir() -> Path:
    """The bundle folder when packaged, the repository root otherwise."""
    if frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[2]


def engine_path() -> Path:
    if frozen():
        return resource_dir() / "engine" / ENGINE_EXE
    return resource_dir() / "image-reader" / "target" / "release" / ENGINE_EXE


def model_path() -> Path:
    if frozen():
        return resource_dir() / "models" / MODEL_NAME
    return resource_dir() / "ai" / "models" / MODEL_NAME


def static_dir() -> Path:
    return resource_dir() / "app" / "static"


def icon_path() -> Path:
    return resource_dir() / "app" / "photoholic.ico"


def user_data_dir() -> Path:
    """Per-user folder for the app's results and logs (``%LOCALAPPDATA%\\Photoholic``)."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "Photoholic"


def runs_dir() -> Path:
    return user_data_dir() / "runs"


def logs_dir() -> Path:
    return user_data_dir() / "logs"


def run_dir_for(folder: Path) -> Path:
    """Stable results folder for a photo folder: ``runs/<name>-<8-char hash of its path>``.

    Nothing is written into the photo folder itself (it may be nearly full or read-only).
    """
    folder = Path(folder).resolve()
    key = str(folder).lower() if os.name == "nt" else str(folder)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    label = folder.name or folder.drive.rstrip(":\\/") or "photos"
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")[:40] or "photos"
    return runs_dir() / f"{name}-{digest}"
