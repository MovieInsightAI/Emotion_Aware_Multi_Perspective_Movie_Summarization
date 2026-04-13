"""
evaluation_plus/__init__.py
=============================
OCP-additive AAAI-grade evaluation suite package.

Exposes the main evaluation tools used by wrappers/enhanced_pipeline.py
and streamlit_app.py.
"""
from evaluation_plus.evaluation_suite import (
    compute_enhanced_emotion_metrics,
    BaselineComparator,
    AblationEvaluator,
    generate_human_eval_template,
)

__all__ = [
    "compute_enhanced_emotion_metrics",
    "BaselineComparator",
    "AblationEvaluator",
    "generate_human_eval_template",
]
