"""
metrics_plus/__init__.py
==========================
OCP-additive extended metrics package.

This package provides supplementary metrics not covered by the base
evaluation_plus/evaluation_suite.py, including:
  - Temporal consistency scoring (arc smoothness over time)
  - Cross-modal agreement metrics (audio vs subtitle emotion alignment)
  - Narrative coherence scores (causal edge density, DAG validity)
  - Perspective orthogonality loss (from MultiPerspectiveEmbedder)

Currently the primary metric implementations live in:
  evaluation_plus/evaluation_suite.py  -> compute_enhanced_emotion_metrics()
  calibration/confidence/calibration_layer.py -> diagnose_calibration()
  research_layers/causal_graph/causal_narrative_model.py -> graph density

Extended metric modules will be registered here as the research matures.
"""
from __future__ import annotations

from typing import Dict, List


def temporal_consistency(emotion_arc: List[Dict[str, float]]) -> float:
    """
    Compute temporal consistency of an emotion arc.

    Measures how smoothly emotion distributions change between consecutive
    scenes. A score of 1.0 = perfectly smooth; 0.0 = maximally erratic.

    Args:
        emotion_arc: List of per-scene emotion dicts, one per scene.

    Returns:
        float in [0, 1] — higher is smoother/more consistent.
    """
    if len(emotion_arc) < 2:
        return 1.0
    try:
        import numpy as np
        labels = list(emotion_arc[0].keys())
        deltas = []
        for i in range(1, len(emotion_arc)):
            prev = np.array([emotion_arc[i - 1].get(k, 0.0) for k in labels])
            curr = np.array([emotion_arc[i].get(k, 0.0) for k in labels])
            deltas.append(float(np.linalg.norm(curr - prev)))
        mean_delta = float(np.mean(deltas))
        # Normalize: max L2 distance between two unit distributions is ~sqrt(2)
        return float(max(0.0, 1.0 - mean_delta / 1.4142))
    except ImportError:
        return 0.5  # neutral fallback if numpy unavailable


def cross_modal_agreement(
    audio_scores: List[Dict[str, float]],
    subtitle_scores: List[Dict[str, float]],
) -> float:
    """
    Compute mean cross-modal emotion agreement between audio and subtitle streams.

    Uses 1 - JSD (Jensen-Shannon Divergence) as a similarity measure,
    averaged over all scenes where both modalities are present.

    Args:
        audio_scores:    Per-scene audio emotion dicts.
        subtitle_scores: Per-scene subtitle emotion dicts.

    Returns:
        float in [0, 1] — higher means audio and subtitles agree more.
    """
    if not audio_scores or not subtitle_scores:
        return 0.5
    try:
        import numpy as np
        labels = list(audio_scores[0].keys())
        n = min(len(audio_scores), len(subtitle_scores))
        agreements = []
        for i in range(n):
            eps = 1e-12
            av = np.array([max(audio_scores[i].get(k, 0.0), eps) for k in labels])
            sv = np.array([max(subtitle_scores[i].get(k, 0.0), eps) for k in labels])
            av /= av.sum()
            sv /= sv.sum()
            m = 0.5 * (av + sv)
            jsd = 0.5 * np.sum(av * np.log(av / m)) + 0.5 * np.sum(sv * np.log(sv / m))
            agreements.append(float(max(0.0, 1.0 - float(jsd))))
        return float(sum(agreements) / len(agreements))
    except ImportError:
        return 0.5


def narrative_coherence(causal_edges: List[Dict], n_scenes: int) -> float:
    """
    Compute narrative coherence from the causal graph edge density.

    A well-connected causal DAG (many causal links between scenes) indicates
    higher narrative coherence. Normalized by max possible edges.

    Args:
        causal_edges: List of edge dicts with keys 'src', 'dst', 'weight'.
        n_scenes:     Total number of scenes in the narrative.

    Returns:
        float in [0, 1] — higher means more causally connected narrative.
    """
    if n_scenes < 2:
        return 1.0
    max_edges = n_scenes * (n_scenes - 1) / 2
    actual = len(causal_edges)
    return float(min(1.0, actual / max(1, max_edges)))


__all__ = ["temporal_consistency", "cross_modal_agreement", "narrative_coherence"]
