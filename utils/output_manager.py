"""
utils/output_manager.py
=========================
Manages saving and organizing pipeline outputs to the outputs/ directory.

Keeps all file I/O logic out of the UI and adapter layers.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_OUTPUTS_ROOT = Path(__file__).resolve().parents[1] / "outputs"


def ensure_outputs_dirs() -> None:
    """Create all required output subdirectories."""
    for sub in ["sessions", "summaries", "evaluation", "keyframes"]:
        (_OUTPUTS_ROOT / sub).mkdir(parents=True, exist_ok=True)


def save_summary_json(
    session_id: str,
    perspective_summaries: Dict[str, str],
    fused_summary: str,
    emotion_distribution: Dict[str, float],
    dominant_emotion: str,
) -> str:
    """Save perspective summaries and fusion output as JSON. Returns the file path."""
    payload = {
        "session_id": session_id,
        "generated_at": datetime.now().isoformat(),
        "dominant_emotion": dominant_emotion,
        "emotion_distribution": emotion_distribution,
        "fused_summary": fused_summary,
        "perspective_summaries": perspective_summaries,
    }
    out_dir = _OUTPUTS_ROOT / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"summary_{session_id}.json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info("Summary saved to %s", path)
    except Exception as exc:
        logger.warning("Could not save summary: %s", exc)
    return str(path)


def save_scene_metadata_json(
    session_id: str,
    scene_records: List[Any],
) -> str:
    """Save scene metadata as JSON. Returns the file path."""
    payload = []
    for scene in scene_records:
        payload.append({
            "scene_id": scene.scene_id,
            "start_time": scene.start_time,
            "end_time": scene.end_time,
            "duration": scene.duration,
            "keyframe_path": scene.keyframe_path or "",
            "visual_embedding_sample": scene.visual_embedding_sample,
        })
    out_dir = _OUTPUTS_ROOT / "sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"scenes_{session_id}.json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception as exc:
        logger.warning("Could not save scene metadata: %s", exc)
    return str(path)


def save_emotion_csv(
    session_id: str,
    emotion_records: List[Any],
) -> str:
    """Save emotion records as CSV. Returns the file path."""
    try:
        import csv
        out_dir = _OUTPUTS_ROOT / "sessions"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"emotions_{session_id}.csv"
        if not emotion_records:
            return str(path)
        fieldnames = ["scene_id", "top_emotion"] + list(emotion_records[0].scores.keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for rec in emotion_records:
                row = {"scene_id": rec.scene_id, "top_emotion": rec.top_emotion}
                row.update(rec.scores)
                writer.writerow(row)
        logger.info("Emotion CSV saved to %s", path)
        return str(path)
    except Exception as exc:
        logger.warning("Could not save emotion CSV: %s", exc)
        return ""


def save_evaluation_json(
    session_id: str,
    evaluation_report: Any,
) -> str:
    """Save evaluation report as JSON. Returns the file path."""
    out_dir = _OUTPUTS_ROOT / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_{session_id}.json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(evaluation_report.raw_metrics, fh, indent=2, default=str)
        logger.info("Evaluation report saved to %s", path)
    except Exception as exc:
        logger.warning("Could not save evaluation report: %s", exc)
    return str(path)
