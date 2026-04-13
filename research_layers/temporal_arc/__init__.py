"""
research_layers/temporal_arc/__init__.py
==========================================
Temporal emotion arc modeling sub-package.
"""
from research_layers.temporal_arc.emotion_arc_model import (
    compute_emotion_arc,
    ArcResult,
)

__all__ = ["compute_emotion_arc", "ArcResult"]
