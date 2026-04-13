"""
integration/adapters/video_adapter.py
=======================================
Adapter that wraps person1_video_module without modifying any of its files.

OCP compliance:
  - person1_video_module/* files are UNTOUCHED.
  - This adapter extends IVideoAnalyser by calling into the legacy module.
  - If the legacy module is unavailable, it falls back gracefully using
    pre-existing sample data already present in the project.
"""

from __future__ import annotations

import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

from integration.interfaces.base_interfaces import IVideoAnalyser, SceneRecord

logger = logging.getLogger(__name__)

# Absolute path to the legacy video module
_LEGACY_VIDEO_MODULE_DIR = Path(__file__).resolve().parents[2] / "person1_video_module"
_SAMPLE_METADATA = _LEGACY_VIDEO_MODULE_DIR / "data" / "outputs" / "scene_metadata.json"
_KEYFRAMES_DIR = _LEGACY_VIDEO_MODULE_DIR / "data" / "keyframes"


class VideoModuleAdapter(IVideoAnalyser):
    """
    Adapts person1_video_module into the IVideoAnalyser interface.

    Strategy:
      1. Attempt to import scene_detector, keyframe_extractor, feature_extractor
         from the legacy src/ package by temporarily adding its parent to sys.path.
      2. If import succeeds, run the full live pipeline.
      3. If import fails or video processing fails, fall back to the pre-existing
         scene_metadata.json and keyframe images already in the project.
    """

    def __init__(self):
        self._live_pipeline_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check whether the live pipeline can run."""
        if self._live_pipeline_available is not None:
            return self._live_pipeline_available

        try:
            self._inject_legacy_path()
            import scenedetect  # noqa: F401
            import cv2  # noqa: F401
            import torch  # noqa: F401
            self._live_pipeline_available = True
        except ImportError as exc:
            logger.warning(
                "Video module live pipeline unavailable (%s). "
                "Will use pre-existing sample data.",
                exc,
            )
            self._live_pipeline_available = False

        return self._live_pipeline_available

    # ------------------------------------------------------------------
    # IVideoAnalyser implementation
    # ------------------------------------------------------------------

    def analyse(self, video_path: str, output_dir: str) -> List[SceneRecord]:
        """
        Run video analysis and return canonical SceneRecord list.

        Falls back to bundled sample data if the live pipeline is unavailable.
        """
        if self.is_available() and video_path and Path(video_path).exists():
            try:
                return self._run_live_pipeline(video_path, output_dir)
            except Exception as exc:
                logger.warning(
                    "Live video pipeline failed (%s). Falling back to sample data.", exc
                )

        return self._load_sample_data()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inject_legacy_path(self):
        """Add the legacy video module to sys.path so its src/ package is importable."""
        legacy_str = str(_LEGACY_VIDEO_MODULE_DIR)
        if legacy_str not in sys.path:
            sys.path.insert(0, legacy_str)

    def _run_live_pipeline(self, video_path: str, output_dir: str) -> List[SceneRecord]:
        """Run the unmodified legacy pipeline components."""
        self._inject_legacy_path()

        # Import legacy modules — they remain untouched
        from src.scene_detector import get_scene_timestamps  # type: ignore
        from src.keyframe_extractor import extract_middle_keyframe  # type: ignore
        from src.feature_extractor import VisualFeatureExtractor  # type: ignore

        keyframes_out = os.path.join(output_dir, "keyframes")
        os.makedirs(keyframes_out, exist_ok=True)

        feature_extractor = VisualFeatureExtractor()
        raw_scenes = get_scene_timestamps(video_path)

        records: List[SceneRecord] = []
        for raw in raw_scenes:
            kf_path = extract_middle_keyframe(video_path, raw, output_dir=keyframes_out)
            features = feature_extractor.extract_features(kf_path)
            compact = features[:10] if features else []

            records.append(
                SceneRecord(
                    scene_id=raw["scene_id"],
                    start_time=round(raw["start_time"], 3),
                    end_time=round(raw["end_time"], 3),
                    keyframe_path=kf_path,
                    visual_embedding_sample=compact,
                )
            )

        return records

    def _load_sample_data(self) -> List[SceneRecord]:
        """
        Load the pre-existing scene_metadata.json bundled with the project.

        This path is preserved exactly as the legacy project left it.
        """
        if not _SAMPLE_METADATA.exists():
            logger.error("Sample metadata not found at %s", _SAMPLE_METADATA)
            return []

        with open(_SAMPLE_METADATA, encoding="utf-8") as fh:
            raw_list = json.load(fh)

        records: List[SceneRecord] = []
        for item in raw_list:
            # Resolve keyframe path relative to the project root
            kf_raw = item.get("keyframe_path", "")
            kf_resolved = self._resolve_keyframe_path(kf_raw)

            records.append(
                SceneRecord(
                    scene_id=item.get("scene_id", 0),
                    start_time=item.get("start_time", 0.0),
                    end_time=item.get("end_time", 0.0),
                    keyframe_path=kf_resolved,
                    visual_embedding_sample=item.get("visual_embedding_sample", []),
                )
            )

        return records

    @staticmethod
    def _resolve_keyframe_path(raw_path: str) -> str:
        """
        Resolve a keyframe path that may use Windows-style separators or relative paths.
        """
        if not raw_path:
            return ""

        # Normalise separators
        normalised = raw_path.replace("\\", "/")

        # Try as-is (absolute)
        if Path(normalised).exists():
            return normalised

        # Try relative to legacy module dir
        candidate = _LEGACY_VIDEO_MODULE_DIR / normalised
        if candidate.exists():
            return str(candidate)

        # Try just the filename in the bundled keyframes dir
        filename = Path(normalised).name
        bundled = _KEYFRAMES_DIR / filename
        if bundled.exists():
            return str(bundled)

        return raw_path  # Return as-is; UI will handle missing gracefully
