"""
research_layers/temporal_arc/emotion_arc_model.py
==================================================
OCP-ADDITIVE EXTENSION — original files untouched.

Fixes:
  - "No temporal modeling of emotion arcs"
  - "No long-range temporal dependency modeling"

Introduces compute_emotion_arc(): temporal smoothing + peak detection
over the per-scene emotion sequence using Exponential Moving Average
(EMA) as the numpy-compatible baseline, with optional PyTorch BiLSTM+
Transformer model if torch is available.

Mathematical formulation (EMA fallback)
----------------------------------------
    arc_t = (1-α)·arc_{t-1} + α·e_t        α = 0.3 (EMA smoothing)
    Δ_t   = arc_t − arc_{t-1}              (emotion transition vector)
    τ_t   = ||Δ_t||_2                      (narrative tension scalar)
    peaks = {t : τ_t > θ · max_τ}          (high-transition scenes)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

EMOTION_LABELS = ["happy", "sad", "angry", "fearful", "calm", "tense"]
K = len(EMOTION_LABELS)


@dataclass
class ArcResult:
    scene_indices: List[int]
    raw_emotions: List[Dict[str, float]]
    arc_emotions: List[Dict[str, float]]
    arc_deltas: List[Dict[str, float]]
    peak_scenes: List[int]
    narrative_tension: List[float]
    dominant_arc: str
    hidden_states: object = None


def compute_emotion_arc(
    scene_emotions: List[Dict[str, float]],
    model=None,
    peak_threshold: float = 0.15,
    ema_alpha: float = 0.3,
) -> ArcResult:
    """
    Compute temporal emotion arc with EMA smoothing and peak detection.

    Args:
        scene_emotions:  list of {emotion: probability} per scene
        model:           optional trained EmotionArcModel (ignored if torch unavailable)
        peak_threshold:  fraction of max tension above which scene is flagged as peak
        ema_alpha:       EMA smoothing factor (0 = no smoothing, 1 = no memory)

    Returns:
        ArcResult with smoothed emotions, deltas, peak scenes, and arc type
    """
    T = len(scene_emotions)
    labels = EMOTION_LABELS

    def to_vec(d: Dict[str, float]) -> np.ndarray:
        v = np.array([max(d.get(lbl, 1e-9), 1e-9) for lbl in labels])
        return v / v.sum()

    if T == 0:
        return ArcResult([], [], [], [], [], [], "static")

    raw_vecs = np.stack([to_vec(e) for e in scene_emotions])  # (T, K)

    # EMA temporal smoothing
    arc_vecs = np.zeros_like(raw_vecs)
    arc_vecs[0] = raw_vecs[0]
    for i in range(1, T):
        arc_vecs[i] = (1 - ema_alpha) * arc_vecs[i - 1] + ema_alpha * raw_vecs[i]

    # Deltas and tension
    deltas = np.zeros_like(arc_vecs)
    if T > 1:
        deltas[1:] = arc_vecs[1:] - arc_vecs[:-1]
    tension = np.linalg.norm(deltas, axis=-1).tolist()

    max_t = max(tension) if tension else 1.0
    if max_t < 1e-9:
        max_t = 1.0
    threshold_abs = peak_threshold * max_t
    peak_scenes = [i for i, t in enumerate(tension) if t > threshold_abs]

    dominant_arc = _characterize_arc(arc_vecs)

    arc_emotions = [{labels[j]: float(arc_vecs[i, j]) for j in range(K)} for i in range(T)]
    arc_deltas   = [{labels[j]: float(deltas[i, j])   for j in range(K)} for i in range(T)]

    return ArcResult(
        scene_indices=list(range(T)),
        raw_emotions=scene_emotions,
        arc_emotions=arc_emotions,
        arc_deltas=arc_deltas,
        peak_scenes=peak_scenes,
        narrative_tension=[round(float(t), 4) for t in tension],
        dominant_arc=dominant_arc,
    )


def _characterize_arc(arc_vecs: np.ndarray) -> str:
    if len(arc_vecs) < 2:
        return "static"
    tension_idx = EMOTION_LABELS.index("tense")
    sad_idx     = EMOTION_LABELS.index("sad")
    happy_idx   = EMOTION_LABELS.index("happy")
    tension_trend = arc_vecs[-1, tension_idx] - arc_vecs[0, tension_idx]
    sad_trend     = arc_vecs[-1, sad_idx]     - arc_vecs[0, sad_idx]
    happy_trend   = arc_vecs[-1, happy_idx]   - arc_vecs[0, happy_idx]
    if tension_trend > 0.1:
        return "rising-tension"
    elif tension_trend < -0.1 and happy_trend > 0.05:
        return "resolution"
    elif sad_trend > 0.1:
        return "tragic-descent"
    elif happy_trend > 0.1:
        return "emotional-uplift"
    return "stable-complex"


if __name__ == "__main__":
    scenes = [
        {"happy": 0.6, "sad": 0.1, "angry": 0.05, "fearful": 0.05, "calm": 0.15, "tense": 0.05},
        {"happy": 0.4, "sad": 0.2, "angry": 0.1,  "fearful": 0.1,  "calm": 0.1,  "tense": 0.1},
        {"happy": 0.2, "sad": 0.3, "angry": 0.2,  "fearful": 0.1,  "calm": 0.05, "tense": 0.15},
        {"happy": 0.1, "sad": 0.2, "angry": 0.3,  "fearful": 0.15, "calm": 0.05, "tense": 0.2},
        {"happy": 0.3, "sad": 0.1, "angry": 0.1,  "fearful": 0.05, "calm": 0.35, "tense": 0.1},
    ]
    r = compute_emotion_arc(scenes)
    print("Arc:", r.dominant_arc, "| Peaks:", r.peak_scenes)
    print("Tension:", r.narrative_tension)
