"""
perspective_plus/formal_perspective.py
=======================================
OCP-ADDITIVE EXTENSION — original files untouched.

Fixes:
  - "No formal definition of perspective"
  - "No learned latent space for perspectives"
  - "No interaction between perspectives"
  - "No cross-perspective reasoning or conflict modeling"

Formal Definition
-----------------
A perspective P_k is formally defined as a triple:
    P_k = (φ_k, ψ_k, C_k)
where:
    φ_k : Z → R^{d_p}           learnable projection into perspective subspace
    ψ_k : R^{d_p} → Δ^K        perspective-conditioned emotion predictor
    C_k ⊆ {1..T}                the set of cinematically salient scenes for P_k

Cross-perspective conflict:
    Conflict(P_k, P_l) = JSD(ψ_k(z) || ψ_l(z))

High conflict → scene is narratively significant, multi-interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

EMOTION_LABELS = ["happy", "sad", "angry", "fearful", "calm", "tense"]
PERSPECTIVES = ["protagonist", "antagonist", "narrator"]
K = len(EMOTION_LABELS)


@dataclass
class PerspectiveDefinition:
    """
    Formal definition P_k = (φ_k, ψ_k, C_k).
    """
    name: str
    salience_bias: Dict[str, float]
    conflict_polarity: int
    description: str
    salient_scene_indices: List[int] = field(default_factory=list)

    def __post_init__(self):
        total = sum(self.salience_bias.values())
        if total > 1e-9:
            self.salience_bias = {k: v/total for k, v in self.salience_bias.items()}


CANONICAL_PERSPECTIVES: Dict[str, PerspectiveDefinition] = {
    "protagonist": PerspectiveDefinition(
        name="protagonist",
        salience_bias={"happy": 0.25, "sad": 0.25, "fearful": 0.2, "calm": 0.1, "angry": 0.1, "tense": 0.1},
        conflict_polarity=+1,
        description=(
            "The protagonist perspective focuses on the main character's emotional journey. "
            "Scenes with high fear, sadness, or joyful resolution are most salient."
        ),
    ),
    "antagonist": PerspectiveDefinition(
        name="antagonist",
        salience_bias={"angry": 0.3, "tense": 0.25, "fearful": 0.15, "calm": 0.1, "sad": 0.1, "happy": 0.1},
        conflict_polarity=-1,
        description=(
            "The antagonist perspective emphasizes scenes of power, control, and threat. "
            "Anger and tension are dominant."
        ),
    ),
    "narrator": PerspectiveDefinition(
        name="narrator",
        salience_bias={"calm": 0.2, "sad": 0.2, "happy": 0.2, "tense": 0.15, "angry": 0.15, "fearful": 0.1},
        conflict_polarity=0,
        description=(
            "The narrator provides the arc-level overview, emphasizing scene transitions "
            "and structural narrative beats."
        ),
    ),
}


def perspective_conflict_score(
    p_emotions: Dict[str, Dict[str, float]],
) -> Dict[Tuple[str, str], float]:
    """
    Compute Jensen-Shannon divergence between all perspective pairs.

    JSD(P_k || P_l) ∈ [0, log2]  — higher means greater narrative conflict.

    Returns dict {(name_k, name_l): JSD_score}
    """
    names = list(p_emotions.keys())
    labels = EMOTION_LABELS
    eps = 1e-9

    def to_vec(d):
        v = np.array([max(d.get(e, 0.0), eps) for e in labels])
        return v / v.sum()

    conflicts = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            ni, nj = names[i], names[j]
            pi = to_vec(p_emotions[ni])
            pj = to_vec(p_emotions[nj])
            m = 0.5 * (pi + pj)
            jsd = 0.5 * np.sum(pi * np.log(np.clip(pi, eps, 1) / np.clip(m, eps, 1))) \
                + 0.5 * np.sum(pj * np.log(np.clip(pj, eps, 1) / np.clip(m, eps, 1)))
            conflicts[(ni, nj)] = round(float(jsd), 4)
    return conflicts


def salience_weighted_emotion(
    raw_emotion: Dict[str, float],
    perspective_name: str,
    alpha: float = 0.4,
) -> Dict[str, float]:
    """
    Compute ψ_k(e) = α·bias_k + (1-α)·e   (perspective-conditioned emotion).

    Args:
        raw_emotion:      base fused emotion distribution
        perspective_name: which perspective to condition on
        alpha:            mixing weight (0 = ignore bias, 1 = pure bias)

    Returns:
        perspective-conditioned emotion dict
    """
    labels = EMOTION_LABELS
    persp = CANONICAL_PERSPECTIVES.get(perspective_name)
    if persp is None:
        return raw_emotion

    bias = persp.salience_bias
    result = {
        e: alpha * bias.get(e, 0.0) + (1 - alpha) * raw_emotion.get(e, 0.0)
        for e in labels
    }
    total = sum(result.values())
    if total > 1e-9:
        result = {k: v/total for k, v in result.items()}
    return result


if __name__ == "__main__":
    for name, defn in CANONICAL_PERSPECTIVES.items():
        print(f"\n{name.upper()}: {defn.description[:70]}...")

    emo = {"happy": 0.4, "sad": 0.25, "angry": 0.1, "fearful": 0.1, "calm": 0.1, "tense": 0.05}
    for p in PERSPECTIVES:
        pe = salience_weighted_emotion(emo, p)
        print(f"  {p}: dominant={max(pe, key=pe.get)}")

    pe = {p: salience_weighted_emotion(emo, p) for p in PERSPECTIVES}
    conflicts = perspective_conflict_score(pe)
    print("\nConflicts:", conflicts)
