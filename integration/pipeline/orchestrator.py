"""
integration/pipeline/orchestrator.py
======================================
Central pipeline orchestrator. Coordinates all four stages:
  1. Video Analysis     → IVideoAnalyser
  2. Emotion Analysis   → IEmotionAnalyser
  3. Summary Generation → ISummaryGenerator
  4. Fusion             → IFusionEngine
  5. Evaluation         → IEvaluator (optional)

OCP compliance:
  - Orchestrator only depends on abstract interfaces, never concrete classes.
  - Concrete implementations are injected via ServiceRegistry.
  - No legacy files are imported or modified here.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Callable, List, Optional

from integration.interfaces.base_interfaces import PipelineResult
from integration.registry.service_registry import ServiceRegistry

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]  # (message, fraction 0-1)


class PipelineOrchestrator:
    """
    Orchestrates the full multimodal movie summarization pipeline.

    All concrete services are injected through the ServiceRegistry, keeping
    this class agnostic to implementation details (OCP).
    """

    def __init__(self, registry: ServiceRegistry, output_root: str = "outputs"):
        self._registry = registry
        self._output_root = Path(output_root)
        self._output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        video_path: str,
        subtitle_path: Optional[str] = None,
        run_evaluation: bool = True,
        reference_summary: Optional[str] = None,
        perspectives: Optional[List[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> PipelineResult:
        """
        Execute the full pipeline and return a PipelineResult.

        Parameters
        ----------
        video_path : str
            Path to the input video file (may be empty to trigger sample-data fallback).
        subtitle_path : Optional[str]
            Path to an .srt subtitle file.
        run_evaluation : bool
            Whether to run the evaluation stage.
        reference_summary : Optional[str]
            Human reference for ROUGE/BLEU computation.
        perspectives : Optional[List[str]]
            Subset of perspectives to generate (default: all three).
        progress_callback : Optional[ProgressCallback]
            Called with (message, fraction) to update a progress bar.
        """
        session_id = str(uuid.uuid4())[:8]
        session_dir = self._output_root / f"session_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        result = PipelineResult(session_id=session_id)
        log = result.processing_log

        def _progress(msg: str, frac: float):
            log.append(f"[{frac*100:.0f}%] {msg}")
            logger.info(msg)
            if progress_callback:
                progress_callback(msg, frac)

        try:
            # ── Stage 1: Video Analysis ───────────────────────────────
            _progress("🎬 Stage 1/5 — Video analysis: scene detection + keyframe extraction", 0.05)
            t0 = time.time()

            video_analyser = self._registry.get_video_analyser()
            scene_records = video_analyser.analyse(
                video_path=video_path,
                output_dir=str(session_dir / "keyframes"),
            )
            result.scenes = scene_records
            _progress(
                f"✅ Video analysis complete — {len(scene_records)} scenes detected "
                f"({time.time()-t0:.1f}s)",
                0.25,
            )

            if not scene_records:
                raise RuntimeError(
                    "Video analysis returned zero scenes. "
                    "Please upload a valid video file."
                )

            # ── Stage 2: Emotion Analysis ─────────────────────────────
            _progress("🎭 Stage 2/5 — Emotion analysis: audio + subtitle inference", 0.28)
            t0 = time.time()

            emotion_analyser = self._registry.get_emotion_analyser()
            emotion_records = emotion_analyser.analyse(
                video_path=video_path,
                scene_records=scene_records,
                subtitle_path=subtitle_path,
                output_dir=str(session_dir),
            )
            result.emotions = emotion_records
            _progress(
                f"✅ Emotion analysis complete — {len(emotion_records)} scenes scored "
                f"({time.time()-t0:.1f}s)",
                0.50,
            )

            # ── Stage 3: Summary Generation ───────────────────────────
            _progress(
                "📖 Stage 3/5 — CRGNN: multi-perspective narrative summary generation", 0.53
            )
            t0 = time.time()

            summary_generator = self._registry.get_summary_generator()
            summary_record = summary_generator.generate(
                scene_records=scene_records,
                emotion_records=emotion_records,
                subtitle_path=subtitle_path,
                output_dir=str(session_dir),
                perspectives=perspectives,
            )
            result.summary = summary_record
            _progress(
                f"✅ Summaries generated — dominant emotion: {summary_record.dominant_emotion} "
                f"({time.time()-t0:.1f}s)",
                0.72,
            )

            # ── Stage 4: Fusion ───────────────────────────────────────
            _progress(
                "🔀 Stage 4/5 — Multimodal fusion: merging video + emotion + summary signals",
                0.75,
            )
            t0 = time.time()

            fusion_engine = self._registry.get_fusion_engine()
            fused_output = fusion_engine.fuse(
                scene_records=scene_records,
                emotion_records=emotion_records,
                summary_record=summary_record,
            )
            result.fused = fused_output
            _progress(
                f"✅ Fusion complete ({time.time()-t0:.1f}s)", 0.88
            )

            # ── Stage 5: Evaluation ───────────────────────────────────
            if run_evaluation:
                _progress("📊 Stage 5/5 — Evaluation metrics computation", 0.90)
                t0 = time.time()

                evaluator = self._registry.get_evaluator()
                evaluation_report = evaluator.evaluate(
                    fused_output=fused_output,
                    reference_summary=reference_summary,
                    output_dir=str(session_dir / "evaluation"),
                )
                result.evaluation = evaluation_report
                _progress(
                    f"✅ Evaluation complete ({time.time()-t0:.1f}s)", 0.98
                )
            else:
                _progress("⏩ Evaluation skipped (disabled by user)", 0.98)

            result.success = True
            _progress(
                f"🎉 Pipeline complete — session {session_id}", 1.0
            )

        except Exception as exc:
            result.success = False
            result.error_message = str(exc)
            log.append(f"[ERROR] {exc}")
            logger.exception("Pipeline failed: %s", exc)

        return result
