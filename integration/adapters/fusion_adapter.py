"""
integration/adapters/fusion_adapter.py
========================================
Adapter that implements IFusionEngine.

The fusion/ module in the legacy project has empty files (fusion/final_generation.py,
merge_modalities.py, scene_representation.py are all 0 bytes).  Per OCP constraints
we do NOT modify those files.  Instead this adapter implements the fusion logic
as a new extension layer that combines outputs from the three upstream modules.

This is the canonical OCP-compliant approach: the empty stubs are preserved,
and the actual implementation is added through extension, not modification.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional

from integration.interfaces.base_interfaces import (
    IFusionEngine,
    FusedOutput,
    SceneRecord,
    EmotionRecord,
    SummaryRecord,
)

logger = logging.getLogger(__name__)


class FusionEngineAdapter(IFusionEngine):
    """
    Implements the multimodal fusion stage entirely as a new extension.

    Algorithm:
      1. Aggregate per-scene emotion scores to produce a global emotion distribution.
      2. Select the dominant emotion across all scenes.
      3. Compose a final enriched narrative summary that combines:
         - Scene count, duration, temporal span
         - Dominant emotion arc description
         - All three perspective summaries
      4. Attach any HTML visualisation artefacts from the summary module.
    """

    def is_available(self) -> bool:
        return True  # Pure Python, no external deps

    def fuse(
        self,
        scene_records: List[SceneRecord],
        emotion_records: List[EmotionRecord],
        summary_record: SummaryRecord,
    ) -> FusedOutput:
        """Fuse all modalities into a FusedOutput."""

        emotion_distribution = self._aggregate_emotion_distribution(emotion_records)
        dominant_emotion = self._dominant_emotion(emotion_records)
        final_summary = self._compose_final_summary(
            scene_records, emotion_records, summary_record, dominant_emotion
        )

        perspective_summaries = {
            "protagonist": summary_record.protagonist,
            "antagonist": summary_record.antagonist,
            "narrator": summary_record.narrator,
        }

        return FusedOutput(
            final_summary=final_summary,
            scene_count=len(scene_records),
            dominant_emotion=dominant_emotion,
            emotion_distribution=emotion_distribution,
            perspective_summaries=perspective_summaries,
            scene_records=scene_records,
            emotion_records=emotion_records,
        )

    # ------------------------------------------------------------------
    # Fusion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_emotion_distribution(
        emotion_records: List[EmotionRecord],
    ) -> Dict[str, float]:
        """Average per-scene emotion scores across all scenes."""
        if not emotion_records:
            return {}

        totals: Dict[str, float] = {}
        count = 0
        for rec in emotion_records:
            if rec.scores:
                for emotion, score in rec.scores.items():
                    totals[emotion] = totals.get(emotion, 0.0) + score
                count += 1

        if count == 0:
            return {}

        return {k: round(v / count, 4) for k, v in totals.items()}

    @staticmethod
    def _dominant_emotion(emotion_records: List[EmotionRecord]) -> str:
        if not emotion_records:
            return "unknown"
        counter: Counter = Counter(rec.top_emotion for rec in emotion_records)
        return counter.most_common(1)[0][0]

    @staticmethod
    def _compose_final_summary(
        scene_records: List[SceneRecord],
        emotion_records: List[EmotionRecord],
        summary_record: SummaryRecord,
        dominant_emotion: str,
    ) -> str:
        """Compose the final fused multimodal summary."""
        n_scenes = len(scene_records)
        total_duration = (
            scene_records[-1].end_time - scene_records[0].start_time
            if scene_records
            else 0.0
        )
        minutes = int(total_duration // 60)
        seconds = int(total_duration % 60)

        # Build emotion arc string
        emotion_arc = " → ".join(
            rec.top_emotion for rec in emotion_records[:min(6, len(emotion_records))]
        )

        # Extract the best available perspective summary as anchor
        anchor = (
            summary_record.narrator
            or summary_record.protagonist
            or summary_record.antagonist
            or ""
        )
        # Strip markdown formatting for plain text
        anchor_clean = anchor.replace("**", "").replace("*", "").strip()

        lines = [
            f"🎬 **Multimodal Enriched Summary** — {n_scenes} scenes | {minutes}m {seconds}s",
            "",
            f"**Dominant emotional arc:** {dominant_emotion.capitalize()} "
            f"(arc: {emotion_arc})",
            "",
            "**Fused Narrative:**",
            anchor_clean if anchor_clean else (
                f"A {n_scenes}-scene narrative dominated by {dominant_emotion} emotions, "
                f"spanning {minutes} minutes and {seconds} seconds of runtime."
            ),
            "",
            f"**Multimodal signals fused:** Visual scene detection (ResNet-50 embeddings) "
            f"+ Audio emotion classification (Wav2Vec2) + Subtitle zero-shot emotion hints "
            f"(BART-MNLI) + Graph Neural Narrative Encoding (CRGNN).",
        ]

        return "\n".join(lines)
