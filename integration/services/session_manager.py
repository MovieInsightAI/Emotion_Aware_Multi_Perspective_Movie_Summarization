"""
integration/services/session_manager.py
=========================================
Manages pipeline session state, caching, and output persistence.

Responsibilities:
  - Cache PipelineResult objects by session_id so Streamlit reruns
    don't re-execute the full pipeline unnecessarily.
  - Provide a clean API for saving and loading session artefacts.
  - Maintain a session log for display in the UI.

OCP: New session types or storage backends can be added without modifying this file.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from integration.interfaces.base_interfaces import PipelineResult

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Lightweight in-process session store with optional disk persistence.

    Sessions are keyed by session_id (UUID prefix).  The store lives for the
    lifetime of the Streamlit process; results are also serialised to disk so
    they survive page refreshes in the same process.
    """

    def __init__(self, sessions_dir: str = "outputs/sessions"):
        self._sessions_dir = Path(sessions_dir)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, PipelineResult] = {}

    # ------------------------------------------------------------------
    # Store / retrieve
    # ------------------------------------------------------------------

    def store(self, result: PipelineResult) -> None:
        """Cache a pipeline result in memory and persist a summary to disk."""
        if not result.session_id:
            return
        self._cache[result.session_id] = result
        self._persist_summary(result)

    def get(self, session_id: str) -> Optional[PipelineResult]:
        """Retrieve a cached pipeline result."""
        return self._cache.get(session_id)

    def list_sessions(self) -> List[Dict]:
        """Return a list of persisted session summaries (most recent first)."""
        summaries: List[Dict] = []
        for path in sorted(
            self._sessions_dir.glob("*.json"), reverse=True
        ):
            try:
                with open(path, encoding="utf-8") as fh:
                    summaries.append(json.load(fh))
            except Exception:
                pass
        return summaries

    def clear(self) -> None:
        """Clear in-memory cache (does not delete disk files)."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Output file helpers
    # ------------------------------------------------------------------

    def get_session_output_dir(self, session_id: str) -> Path:
        return Path("outputs") / f"session_{session_id}"

    def get_keyframe_paths(self, session_id: str) -> List[str]:
        """Return sorted list of keyframe image paths for a session."""
        kf_dir = self.get_session_output_dir(session_id) / "keyframes"
        if not kf_dir.exists():
            return []
        return sorted(str(p) for p in kf_dir.glob("*.jpg"))

    def get_evaluation_report_path(self, session_id: str) -> Optional[str]:
        """Return path to evaluation_report.json if it exists."""
        path = self.get_session_output_dir(session_id) / "evaluation" / "evaluation_report.json"
        return str(path) if path.exists() else None

    def get_html_artefact(self, session_id: str, filename: str) -> Optional[str]:
        """Read an HTML artefact from the session output dir."""
        path = self.get_session_output_dir(session_id) / filename
        if path.exists():
            try:
                with open(path, encoding="utf-8") as fh:
                    return fh.read()
            except Exception:
                pass
        # Fall back to sample_outputs from the legacy module
        legacy_sample = (
            Path(__file__).resolve().parents[2]
            / "person3_summary_module"
            / "sample_outputs"
            / filename
        )
        if legacy_sample.exists():
            try:
                with open(legacy_sample, encoding="utf-8") as fh:
                    return fh.read()
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _persist_summary(self, result: PipelineResult) -> None:
        summary = {
            "session_id": result.session_id,
            "timestamp": datetime.now().isoformat(),
            "success": result.success,
            "scene_count": len(result.scenes),
            "emotion_count": len(result.emotions),
            "dominant_emotion": (
                result.fused.dominant_emotion if result.fused else "unknown"
            ),
            "error": result.error_message,
        }
        path = self._sessions_dir / f"{result.session_id}.json"
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
        except Exception as exc:
            logger.warning("Could not persist session summary: %s", exc)
