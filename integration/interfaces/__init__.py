"""
integration/interfaces/__init__.py
===================================
Abstract base interfaces for all integration adapters.
OCP: These interfaces are open for extension, closed for modification.
No original project files are touched.
"""

from .base_interfaces import (
    IVideoAnalyser,
    IEmotionAnalyser,
    ISummaryGenerator,
    IFusionEngine,
    IEvaluator,
    PipelineResult,
    SceneRecord,
    EmotionRecord,
    SummaryRecord,
    FusedOutput,
    EvaluationReport,
)

__all__ = [
    "IVideoAnalyser",
    "IEmotionAnalyser",
    "ISummaryGenerator",
    "IFusionEngine",
    "IEvaluator",
    "PipelineResult",
    "SceneRecord",
    "EmotionRecord",
    "SummaryRecord",
    "FusedOutput",
    "EvaluationReport",
]
