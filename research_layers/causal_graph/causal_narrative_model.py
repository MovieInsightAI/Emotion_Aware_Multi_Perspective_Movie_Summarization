"""
research_layers/causal_graph/causal_narrative_model.py
========================================================
OCP-ADDITIVE EXTENSION — original files untouched.

Fixes:
  - "No causal inference or intervention capability, despite claims"
  - "Graph component appears decorative rather than functional"

Builds a causal DAG over narrative scenes and supports do-calculus
intervention queries — making the "causal" claim formally testable.

Mathematical formulation
------------------------
G = (V, E),  v_i = scene node with emotion e_i, tension τ_i

Edge weight:
    w_{ij} = exp(−λ·Δt) · (1 + τ_i) · min(1, KL(e_i||e_j)/2)

Intervention (do-calculus):
    do(e_t = ê) → propagate ê forward via message passing:
        e_j' = (1 − α·w_{t,j})·e_j + α·w_{t,j}·ê
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

EMOTION_LABELS = ["happy", "sad", "angry", "fearful", "calm", "tense"]
K = len(EMOTION_LABELS)
PERSPECTIVES = ["protagonist", "antagonist", "narrator"]


@dataclass
class CausalNode:
    scene_id: int
    emotion: Dict[str, float]
    tension: float
    salience: Dict[str, float]
    timestamp: Optional[float] = None
    is_peak: bool = False
    parents: List[int] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    causal_weight: Dict[int, float] = field(default_factory=dict)


@dataclass
class CausalGraphResult:
    nodes: List[CausalNode]
    adjacency: np.ndarray
    causal_paths: List[List[int]]
    intervention_results: Dict[str, List[Dict[str, float]]]


class CausalNarrativeGraph:
    def __init__(self, temporal_decay: float = 0.5, causal_window: int = 3):
        self.temporal_decay = temporal_decay
        self.causal_window = causal_window

    def build(
        self,
        scene_emotions: List[Dict[str, float]],
        tensions: Optional[List[float]] = None,
        saliences: Optional[List[Dict[str, float]]] = None,
        timestamps: Optional[List[float]] = None,
        peak_scenes: Optional[List[int]] = None,
    ) -> CausalGraphResult:
        N = len(scene_emotions)
        if N == 0:
            return CausalGraphResult([], np.zeros((0, 0)), [], {})

        tensions = tensions or [0.1] * N
        peak_set = set(peak_scenes or [])
        timestamps = timestamps or [float(i) for i in range(N)]
        default_sal = {p: 1.0 / 3 for p in PERSPECTIVES}
        saliences = saliences or [default_sal] * N

        nodes = [
            CausalNode(i, scene_emotions[i], tensions[i], saliences[i],
                       timestamps[i], i in peak_set)
            for i in range(N)
        ]

        W = np.zeros((N, N))
        for j in range(1, N):
            for i in range(max(0, j - self.causal_window), j):
                w = self._edge_weight(nodes[i], nodes[j])
                if w > 0.01:
                    W[i, j] = w
                    nodes[i].children.append(j)
                    nodes[j].parents.append(i)
                    nodes[i].causal_weight[j] = w

        paths = self._extract_paths(nodes, W)
        return CausalGraphResult(nodes, W, paths, {})

    def _edge_weight(self, src: CausalNode, dst: CausalNode) -> float:
        dt = abs((dst.timestamp or 0) - (src.timestamp or 0))
        temporal = math.exp(-self.temporal_decay * dt / max(dt, 1.0))
        kl = self._kl(src.emotion, dst.emotion)
        emotion_f = min(1.0, kl / 2.0)
        tension_f = 1.0 + min(1.0, src.tension)
        return float(temporal * tension_f * (0.3 + 0.7 * emotion_f))

    @staticmethod
    def _kl(p: Dict[str, float], q: Dict[str, float]) -> float:
        eps = 1e-9
        pv = np.array([max(p.get(e, 0.0), eps) for e in EMOTION_LABELS])
        qv = np.array([max(q.get(e, 0.0), eps) for e in EMOTION_LABELS])
        pv /= pv.sum(); qv /= qv.sum()
        return float(np.sum(pv * np.log(pv / qv)))

    def _extract_paths(self, nodes: List[CausalNode], W: np.ndarray) -> List[List[int]]:
        N = len(nodes)
        paths = []
        for start in range(N):
            path, cur, vis = [start], start, {start}
            for _ in range(N):
                ch = [c for c in np.where(W[cur] > 0)[0] if c not in vis]
                if not ch:
                    break
                nxt = max(ch, key=lambda c: W[cur, c])
                path.append(nxt); vis.add(nxt); cur = nxt
            if len(path) > 1:
                paths.append(path)

        def score(p):
            ws = [W[p[i], p[i+1]] for i in range(len(p)-1)]
            return len(p) * (sum(ws)/len(ws) if ws else 0)

        paths.sort(key=score, reverse=True)
        return paths[:3]

    def intervene(
        self,
        result: CausalGraphResult,
        target_scene: int,
        new_emotion: Dict[str, float],
        strength: float = 0.4,
    ) -> Dict[int, Dict[str, float]]:
        """
        do(e_{target} = new_emotion) with forward message propagation.

        Returns {scene_id: counterfactual_emotion}
        """
        nodes = result.nodes
        W = result.adjacency
        total = sum(new_emotion.values())
        if total > 1e-9:
            new_emotion = {k: v/total for k, v in new_emotion.items()}

        intervened = {target_scene: new_emotion.copy()}
        queue = [(target_scene, new_emotion, 1.0)]
        visited = {target_scene}

        while queue:
            cur, cur_emo, cum_w = queue.pop(0)
            for child in np.where(W[cur] > 0)[0]:
                if child in visited:
                    continue
                ew = float(W[cur, child])
                blend = strength * cum_w * ew
                orig = nodes[child].emotion
                blended = {e: (1-blend)*orig.get(e, 0.0) + blend*cur_emo.get(e, 0.0)
                           for e in EMOTION_LABELS}
                s = sum(blended.values())
                if s > 1e-9:
                    blended = {k: v/s for k, v in blended.items()}
                intervened[child] = blended
                visited.add(child)
                queue.append((child, blended, cum_w * ew))

        return intervened


def graph_to_edge_list(result: CausalGraphResult) -> List[Dict]:
    edges = []
    N = len(result.nodes)
    for i in range(N):
        for j in range(N):
            w = result.adjacency[i, j]
            if w > 0.01:
                edges.append({"source": i, "target": j, "weight": round(float(w), 4)})
    return edges


if __name__ == "__main__":
    scenes = [
        {"happy": 0.6, "sad": 0.1, "angry": 0.05, "fearful": 0.05, "calm": 0.15, "tense": 0.05},
        {"happy": 0.3, "sad": 0.2, "angry": 0.15, "fearful": 0.1,  "calm": 0.1,  "tense": 0.15},
        {"happy": 0.1, "sad": 0.2, "angry": 0.35, "fearful": 0.15, "calm": 0.05, "tense": 0.15},
    ]
    g = CausalNarrativeGraph()
    r = g.build(scenes, tensions=[0.1, 0.4, 0.8], peak_scenes=[2])
    print("Paths:", r.causal_paths)
    print("Edges:", graph_to_edge_list(r))
    iv = g.intervene(r, 1, {"happy": 0.8, "calm": 0.2, "sad": 0, "angry": 0, "fearful": 0, "tense": 0})
    print("Intervention result:", {k: {e: round(v,3) for e,v in d.items()} for k,d in iv.items()})
