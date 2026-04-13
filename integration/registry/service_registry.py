"""
integration/registry/service_registry.py
==========================================
Central registry for all pipeline service adapters.

OCP compliance:
  - New adapters can be registered without modifying existing registry code.
  - Consumers look up services by interface name, not concrete class.
  - The registry is the only place where concrete classes are instantiated.

This follows the Registry pattern (a form of Dependency Injection) which
enables easy swapping of implementations during testing or future extension.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Type

from integration.interfaces.base_interfaces import (
    IVideoAnalyser,
    IEmotionAnalyser,
    ISummaryGenerator,
    IFusionEngine,
    IEvaluator,
)

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """
    Singleton-style registry that holds one concrete implementation per interface.

    Usage:
        registry = ServiceRegistry.build_default()
        video_analyser = registry.get_video_analyser()
    """

    def __init__(self):
        self._video_analyser: Optional[IVideoAnalyser] = None
        self._emotion_analyser: Optional[IEmotionAnalyser] = None
        self._summary_generator: Optional[ISummaryGenerator] = None
        self._fusion_engine: Optional[IFusionEngine] = None
        self._evaluator: Optional[IEvaluator] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build_default(cls) -> "ServiceRegistry":
        """
        Instantiate the default registry using all standard adapters.

        This is the only place that imports concrete adapter classes,
        keeping the rest of the codebase decoupled from implementation details.
        """
        from integration.adapters.video_adapter import VideoModuleAdapter
        from integration.adapters.emotion_adapter import EmotionModuleAdapter
        from integration.adapters.summary_adapter import SummaryModuleAdapter
        from integration.adapters.fusion_adapter import FusionEngineAdapter
        from integration.adapters.evaluation_adapter import EvaluationAdapter

        registry = cls()
        registry.register_video_analyser(VideoModuleAdapter())
        registry.register_emotion_analyser(EmotionModuleAdapter())
        registry.register_summary_generator(SummaryModuleAdapter())
        registry.register_fusion_engine(FusionEngineAdapter())
        registry.register_evaluator(EvaluationAdapter())

        return registry

    # ------------------------------------------------------------------
    # Registration (open for extension: new adapters can be registered)
    # ------------------------------------------------------------------

    def register_video_analyser(self, impl: IVideoAnalyser) -> None:
        self._video_analyser = impl
        logger.debug("Registered video analyser: %s", type(impl).__name__)

    def register_emotion_analyser(self, impl: IEmotionAnalyser) -> None:
        self._emotion_analyser = impl
        logger.debug("Registered emotion analyser: %s", type(impl).__name__)

    def register_summary_generator(self, impl: ISummaryGenerator) -> None:
        self._summary_generator = impl
        logger.debug("Registered summary generator: %s", type(impl).__name__)

    def register_fusion_engine(self, impl: IFusionEngine) -> None:
        self._fusion_engine = impl
        logger.debug("Registered fusion engine: %s", type(impl).__name__)

    def register_evaluator(self, impl: IEvaluator) -> None:
        self._evaluator = impl
        logger.debug("Registered evaluator: %s", type(impl).__name__)

    # ------------------------------------------------------------------
    # Lookup (typed, safe)
    # ------------------------------------------------------------------

    def get_video_analyser(self) -> IVideoAnalyser:
        if self._video_analyser is None:
            raise RuntimeError("No IVideoAnalyser registered.")
        return self._video_analyser

    def get_emotion_analyser(self) -> IEmotionAnalyser:
        if self._emotion_analyser is None:
            raise RuntimeError("No IEmotionAnalyser registered.")
        return self._emotion_analyser

    def get_summary_generator(self) -> ISummaryGenerator:
        if self._summary_generator is None:
            raise RuntimeError("No ISummaryGenerator registered.")
        return self._summary_generator

    def get_fusion_engine(self) -> IFusionEngine:
        if self._fusion_engine is None:
            raise RuntimeError("No IFusionEngine registered.")
        return self._fusion_engine

    def get_evaluator(self) -> IEvaluator:
        if self._evaluator is None:
            raise RuntimeError("No IEvaluator registered.")
        return self._evaluator

    def availability_report(self) -> Dict[str, bool]:
        """Return a dict of service name → availability status."""
        return {
            "video_analyser": self._video_analyser.is_available() if self._video_analyser else False,
            "emotion_analyser": self._emotion_analyser.is_available() if self._emotion_analyser else False,
            "summary_generator": self._summary_generator.is_available() if self._summary_generator else False,
            "fusion_engine": self._fusion_engine.is_available() if self._fusion_engine else False,
            "evaluator": self._evaluator.is_available() if self._evaluator else False,
        }
