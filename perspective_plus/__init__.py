"""
perspective_plus/__init__.py
==============================
OCP-additive formal perspective definition package.
"""
from perspective_plus.formal_perspective import (
    perspective_conflict_score,
    salience_weighted_emotion,
    CANONICAL_PERSPECTIVES,
    PERSPECTIVES,
    PerspectiveDefinition,
)

__all__ = [
    "perspective_conflict_score",
    "salience_weighted_emotion",
    "CANONICAL_PERSPECTIVES",
    "PERSPECTIVES",
    "PerspectiveDefinition",
]
