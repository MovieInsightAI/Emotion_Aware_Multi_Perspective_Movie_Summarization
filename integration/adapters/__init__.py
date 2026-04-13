"""
integration/adapters/__init__.py
"""
from .video_adapter import VideoModuleAdapter
from .emotion_adapter import EmotionModuleAdapter
from .summary_adapter import SummaryModuleAdapter
from .fusion_adapter import FusionEngineAdapter
from .evaluation_adapter import EvaluationAdapter

__all__ = [
    "VideoModuleAdapter",
    "EmotionModuleAdapter",
    "SummaryModuleAdapter",
    "FusionEngineAdapter",
    "EvaluationAdapter",
]
