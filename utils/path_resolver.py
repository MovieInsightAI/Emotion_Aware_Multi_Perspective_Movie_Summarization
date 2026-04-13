"""
utils/path_resolver.py
========================
Utilities for resolving cross-platform, cross-module paths safely.

Handles Windows backslash paths, relative legacy paths, and missing files
without raising exceptions so the UI never crashes on path issues.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Root of the integration layer (where streamlit_app.py lives)
_INTEGRATION_ROOT = Path(__file__).resolve().parents[1]

# Root of the legacy project (one level above integration layer)
_PROJECT_ROOT = _INTEGRATION_ROOT.parent


def resolve_legacy_path(raw_path: str) -> Optional[str]:
    """
    Resolve a path that may come from the legacy project (Windows-style
    separators, relative to the legacy module root, etc.).

    Returns None if the file does not exist after all resolution attempts.
    """
    if not raw_path:
        return None

    normalised = raw_path.replace("\\", "/")

    # Candidate 1: as-is (may already be absolute)
    p = Path(normalised)
    if p.exists():
        return str(p)

    # Candidate 2: relative to project root
    p2 = _PROJECT_ROOT / normalised
    if p2.exists():
        return str(p2)

    # Candidate 3: relative to integration root
    p3 = _INTEGRATION_ROOT / normalised
    if p3.exists():
        return str(p3)

    # Candidate 4: just the filename inside known legacy data dirs
    filename = Path(normalised).name
    for search_dir in [
        _PROJECT_ROOT / "person1_video_module" / "data" / "keyframes",
        _PROJECT_ROOT / "person3_summary_module" / "sample_outputs",
    ]:
        candidate = search_dir / filename
        if candidate.exists():
            return str(candidate)

    return None


def get_bundled_keyframes_dir() -> Path:
    """Return the path to the pre-existing bundled keyframes directory."""
    return _PROJECT_ROOT / "person1_video_module" / "data" / "keyframes"


def get_sample_outputs_dir() -> Path:
    """Return the path to the pre-existing sample outputs directory."""
    return _PROJECT_ROOT / "person3_summary_module" / "sample_outputs"


def get_outputs_root() -> Path:
    """Return the integration-layer outputs root."""
    p = _INTEGRATION_ROOT / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_read_json(path: str | Path) -> Optional[dict]:
    """Read a JSON file, returning None on any error."""
    import json
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def safe_read_html(path: str | Path) -> Optional[str]:
    """Read an HTML file, returning None on any error."""
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        return None
