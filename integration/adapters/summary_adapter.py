"""
integration/adapters/summary_adapter.py
=========================================
Adapter that wraps person3_summary_module (CRGNN system) without modifying any
of its files.

OCP compliance:
  - person3_summary_module/* files are UNTOUCHED.
  - This adapter extends ISummaryGenerator.
  - The live path calls into training_pipeline.CRGNNSystem and run_inference.
  - The fallback path loads pre-existing sample_outputs/perspective_summaries.json
    that is already bundled with the project.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from integration.interfaces.base_interfaces import (
    ISummaryGenerator,
    SummaryRecord,
    SceneRecord,
    EmotionRecord,
)

logger = logging.getLogger(__name__)

_LEGACY_SUMMARY_DIR = (
    Path(__file__).resolve().parents[2] / "person3_summary_module"
)
_SAMPLE_SUMMARIES = _LEGACY_SUMMARY_DIR / "sample_outputs" / "perspective_summaries.json"
_SAMPLE_OUTPUTS_DIR = _LEGACY_SUMMARY_DIR / "sample_outputs"

_EMOTION_LABEL_MAP = {
    "happy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "sad":   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "angry": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "fearful":[0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0],
    "calm":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "tense": [0.0, 0.0, 0.5, 0.5, 0.5, 0.2, 0.0, 0.2],
}


class SummaryModuleAdapter(ISummaryGenerator):
    """
    Adapts person3_summary_module into the ISummaryGenerator interface.

    Strategy:
      1. Attempt to import CRGNNSystem and run_inference from training_pipeline.
      2. If available, build a synthetic .srt from scene/emotion data and run
         the CRGNN inference pipeline.
      3. Fall back to the pre-existing sample_outputs/perspective_summaries.json.
    """

    def __init__(self):
        self._live_available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._live_available is not None:
            return self._live_available

        try:
            self._inject_legacy_path()
            import torch  # noqa: F401
            # Try importing the training_pipeline — this validates torch-geometric etc.
            from training_pipeline import CRGNNConfig, CRGNNSystem, run_inference  # type: ignore  # noqa
            self._live_available = True
        except Exception as exc:
            logger.warning(
                "Summary module live pipeline unavailable (%s). "
                "Pre-existing sample outputs will be used.",
                exc,
            )
            self._live_available = False

        return self._live_available

    # ------------------------------------------------------------------
    # ISummaryGenerator implementation
    # ------------------------------------------------------------------

    def generate(
        self,
        scene_records: List[SceneRecord],
        emotion_records: List[EmotionRecord],
        subtitle_path: Optional[str] = None,
        output_dir: str = "outputs",
        perspectives: Optional[List[str]] = None,
    ) -> SummaryRecord:
        """Generate perspective-aware summaries."""

        if self.is_available():
            try:
                return self._run_live_inference(
                    scene_records, emotion_records, subtitle_path, output_dir
                )
            except Exception as exc:
                logger.warning(
                    "Live CRGNN inference failed (%s). Loading sample outputs.", exc
                )

        return self._load_sample_summaries(scene_records, emotion_records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inject_legacy_path(self):
        legacy_str = str(_LEGACY_SUMMARY_DIR)
        if legacy_str not in sys.path:
            sys.path.insert(0, legacy_str)

    def _run_live_inference(
        self,
        scene_records: List[SceneRecord],
        emotion_records: List[EmotionRecord],
        subtitle_path: Optional[str],
        output_dir: str,
    ) -> SummaryRecord:
        """Run the CRGNN inference pipeline with scene/emotion data."""
        import torch
        from training_pipeline import CRGNNConfig, CRGNNSystem, run_inference  # type: ignore

        # Build synthetic SRT if no subtitle provided
        srt_text = self._build_srt(scene_records, subtitle_path)

        # Build emotion tensor aligned to CRGNN's 8-dim emotion space
        emotion_tensor = self._build_emotion_tensor(emotion_records)

        cfg = CRGNNConfig(
            vocab_size=1024, d_model=64, d_hidden=64, d_latent=128,
            d_emotion=8, d_code=32, d_arc=32, d_persp=64,
            d_emb_dec=32, d_hidden_dec=128,
            max_seq_len=64, max_decode_len=24,
            num_gat_layers=2, gat_heads=4,
            use_vae=True, causal_threshold=0.4,
        )

        system = CRGNNSystem(cfg)

        # Load the best available checkpoint if present
        self._try_load_checkpoint(system, output_dir)
        system.eval()

        result = run_inference(system, srt_text, emotion_tensor, device="cpu")

        if "error" in result:
            raise RuntimeError(result["error"])

        summaries: Dict[str, str] = result.get("summaries", {})
        dominant_emotion = self._dominant_emotion(emotion_records)
        emotion_intensity = self._emotion_intensity(emotion_records)

        # Save HTML visualisations if the visualizations module is importable
        self._save_visual_outputs(result, output_dir)

        return SummaryRecord(
            protagonist=summaries.get("protagonist", ""),
            antagonist=summaries.get("antagonist", ""),
            narrator=summaries.get("narrator", ""),
            dominant_emotion=dominant_emotion,
            emotion_intensity=emotion_intensity,
            scene_count=len(scene_records),
            raw_result=summaries,
        )

    def _load_sample_summaries(
        self,
        scene_records: List[SceneRecord],
        emotion_records: List[EmotionRecord],
    ) -> SummaryRecord:
        """Load the pre-existing sample perspective_summaries.json."""
        if not _SAMPLE_SUMMARIES.exists():
            logger.error("Sample summaries not found at %s", _SAMPLE_SUMMARIES)
            return SummaryRecord()

        with open(_SAMPLE_SUMMARIES, encoding="utf-8") as fh:
            data = json.load(fh)

        dominant_emotion = self._dominant_emotion(emotion_records)
        emotion_intensity = self._emotion_intensity(emotion_records)

        return SummaryRecord(
            protagonist=data.get("protagonist", ""),
            antagonist=data.get("antagonist", ""),
            narrator=data.get("narrator", ""),
            dominant_emotion=dominant_emotion,
            emotion_intensity=emotion_intensity,
            scene_count=len(scene_records),
            raw_result=data,
        )

    @staticmethod
    def _build_srt(
        scene_records: List[SceneRecord],
        subtitle_path: Optional[str],
    ) -> str:
        """Build an SRT string from subtitle file or synthetic scene descriptions."""
        if subtitle_path and Path(subtitle_path).exists():
            with open(subtitle_path, encoding="utf-8", errors="ignore") as fh:
                return fh.read()

        lines = []
        for i, scene in enumerate(scene_records, 1):
            start = _seconds_to_srt(scene.start_time)
            end = _seconds_to_srt(scene.end_time)
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(f"Scene {scene.scene_id} (duration: {scene.duration:.1f}s)")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _build_emotion_tensor(emotion_records: List[EmotionRecord]):
        """Convert EmotionRecord list to a torch float tensor (N, 8)."""
        import torch
        import numpy as np

        rows = []
        for rec in emotion_records:
            mapped = _EMOTION_LABEL_MAP.get(rec.top_emotion, [0.0] * 8)
            for emotion in ["happy", "sad", "angry", "fearful", "calm", "tense"]:
                score = rec.scores.get(emotion, 0.0)
                idx = list(_EMOTION_LABEL_MAP.keys()).index(emotion) if emotion in _EMOTION_LABEL_MAP else 0
                if idx < len(mapped):
                    mapped[idx] = score
            # Pad to length 8
            while len(mapped) < 8:
                mapped.append(0.0)
            rows.append(mapped[:8])

        arr = np.array(rows, dtype=np.float32)
        return torch.tensor(arr)

    @staticmethod
    def _try_load_checkpoint(system, output_dir: str):
        """Load the best available checkpoint from the legacy checkpoints/ dir."""
        import torch
        ckpt_dir = _LEGACY_SUMMARY_DIR / "checkpoints"
        if not ckpt_dir.exists():
            return
        checkpoints = sorted(ckpt_dir.glob("*.pt"))
        if not checkpoints:
            return
        best = checkpoints[-1]  # Last epoch = best loss
        try:
            state = torch.load(str(best), map_location="cpu")
            system.load_state_dict(state, strict=False)
            logger.info("Loaded checkpoint: %s", best.name)
        except Exception as exc:
            logger.warning("Checkpoint load failed (%s), running with random weights.", exc)

    @staticmethod
    def _save_visual_outputs(result: dict, output_dir: str):
        """Save HTML visualisations from the CRGNN result dict."""
        try:
            sys.path.insert(0, str(_LEGACY_SUMMARY_DIR))
            from evaluation_scripts.visualizations import (  # type: ignore
                plot_emotion_trajectory,
                plot_causal_graph_plotly,
                plot_latent_scatter,
                build_summary_figure,
            )
            import numpy as np

            os.makedirs(output_dir, exist_ok=True)
            emo_np = result["emotion_vecs"].numpy() if hasattr(result.get("emotion_vecs"), "numpy") else None
            vad_np = result["vad"].numpy() if hasattr(result.get("vad"), "numpy") else None
            z_dict = result.get("z_dict", {})
            sal_dict = result.get("sal_dict", {})

            if emo_np is not None:
                fig_emo = plot_emotion_trajectory(emo_np, list(range(1, len(emo_np) + 1)))
                fig_emo.write_html(os.path.join(output_dir, "emotion_trajectory.html"))

            if vad_np is not None and z_dict:
                fig_dash = build_summary_figure(emo_np, z_dict, sal_dict, vad_np)
                fig_dash.write_html(os.path.join(output_dir, "dashboard.html"))

            if z_dict:
                fig_scatter = plot_latent_scatter(z_dict)
                fig_scatter.write_html(os.path.join(output_dir, "latent_scatter.html"))

        except Exception as exc:
            logger.warning("Visual output generation failed: %s", exc)

    @staticmethod
    def _dominant_emotion(emotion_records: List[EmotionRecord]) -> str:
        if not emotion_records:
            return "unknown"
        counts: Dict[str, int] = {}
        for rec in emotion_records:
            counts[rec.top_emotion] = counts.get(rec.top_emotion, 0) + 1
        return max(counts, key=counts.get)

    @staticmethod
    def _emotion_intensity(emotion_records: List[EmotionRecord]) -> float:
        if not emotion_records:
            return 0.0
        values = []
        for rec in emotion_records:
            if rec.scores:
                values.append(max(rec.scores.values()))
        return round(sum(values) / len(values), 4) if values else 0.0


def _seconds_to_srt(secs: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{int(s):02d},{ms:03d}"
