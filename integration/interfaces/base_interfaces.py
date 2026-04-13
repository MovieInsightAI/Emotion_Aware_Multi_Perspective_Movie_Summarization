"""
integration/interfaces/base_interfaces.py
==========================================
Abstract base classes (ABCs) that define the contracts for each pipeline stage.

Design principle (OCP):
  - These interfaces are CLOSED for modification.
  - New adapters extend them without altering this file.
  - All adapters that wrap legacy modules must implement these ABCs.

No legacy file is imported or modified here.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data-transfer objects (plain dataclasses, no business logic)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SceneRecord:
    """Canonical representation of one detected scene."""
    scene_id: int
    start_time: float
    end_time: float
    duration: float = 0.0
    keyframe_path: Optional[str] = None
    visual_embedding_sample: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.duration = round(self.end_time - self.start_time, 3)


@dataclass
class EmotionRecord:
    """Canonical representation of emotion scores for one scene."""
    scene_id: int
    top_emotion: str
    scores: Dict[str, float] = field(default_factory=dict)
    subtitle_text: str = ""
    audio_path: Optional[str] = None


@dataclass
class SummaryRecord:
    """Canonical representation of multi-perspective summaries."""
    protagonist: str = ""
    antagonist: str = ""
    narrator: str = ""
    dominant_emotion: str = ""
    emotion_intensity: float = 0.0
    scene_count: int = 0
    raw_result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedOutput:
    """Canonical representation of the final fused multimodal output."""
    final_summary: str = ""
    scene_count: int = 0
    dominant_emotion: str = ""
    emotion_distribution: Dict[str, float] = field(default_factory=dict)
    perspective_summaries: Dict[str, str] = field(default_factory=dict)
    scene_records: List[SceneRecord] = field(default_factory=list)
    emotion_records: List[EmotionRecord] = field(default_factory=list)
    visual_metadata: List[Dict[str, Any]] = field(default_factory=list)
    causal_graph_html: Optional[str] = None
    emotion_trajectory_html: Optional[str] = None
    latent_scatter_html: Optional[str] = None
    dashboard_html: Optional[str] = None


@dataclass
class EvaluationReport:
    """Canonical representation of all evaluation metrics."""
    rouge_scores: Dict[str, float] = field(default_factory=dict)
    bleu_scores: Dict[str, float] = field(default_factory=dict)
    graph_metrics: Dict[str, float] = field(default_factory=dict)
    perspective_divergence: Dict[str, float] = field(default_factory=dict)
    emotion_consistency: Dict[str, float] = field(default_factory=dict)
    latent_quality: Dict[str, float] = field(default_factory=dict)
    raw_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Top-level result container for the full orchestrated pipeline."""
    success: bool = False
    error_message: str = ""
    scenes: List[SceneRecord] = field(default_factory=list)
    emotions: List[EmotionRecord] = field(default_factory=list)
    summary: Optional[SummaryRecord] = None
    fused: Optional[FusedOutput] = None
    evaluation: Optional[EvaluationReport] = None
    session_id: str = ""
    processing_log: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interfaces
# ─────────────────────────────────────────────────────────────────────────────

class IVideoAnalyser(abc.ABC):
    """
    Contract for video analysis: scene detection + keyframe extraction + feature extraction.

    Concrete implementations wrap person1_video_module without modifying it.
    """

    @abc.abstractmethod
    def analyse(self, video_path: str, output_dir: str) -> List[SceneRecord]:
        """
        Analyse a video file and return a list of SceneRecord objects.

        Parameters
        ----------
        video_path : str
            Absolute path to the input video file.
        output_dir : str
            Directory where keyframes and outputs should be written.

        Returns
        -------
        List[SceneRecord]
        """

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if all underlying dependencies are importable and functional."""


class IEmotionAnalyser(abc.ABC):
    """
    Contract for audio/subtitle-based emotion analysis.

    Concrete implementations wrap person2_emotion_module without modifying it.
    """

    @abc.abstractmethod
    def analyse(
        self,
        video_path: str,
        scene_records: List[SceneRecord],
        subtitle_path: Optional[str] = None,
        output_dir: str = "outputs",
    ) -> List[EmotionRecord]:
        """
        Analyse emotions for each scene and return EmotionRecord list.

        Parameters
        ----------
        video_path : str
            Path to the source video.
        scene_records : List[SceneRecord]
            Scene boundary information from the video analyser.
        subtitle_path : Optional[str]
            Path to an .srt file for subtitle-based emotion hints.
        output_dir : str
            Directory for intermediate audio clips.

        Returns
        -------
        List[EmotionRecord]
        """

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if all underlying dependencies are importable and functional."""


class ISummaryGenerator(abc.ABC):
    """
    Contract for multi-perspective, emotion-aware summary generation.

    Concrete implementations wrap person3_summary_module without modifying it.
    """

    @abc.abstractmethod
    def generate(
        self,
        scene_records: List[SceneRecord],
        emotion_records: List[EmotionRecord],
        subtitle_path: Optional[str] = None,
        output_dir: str = "outputs",
        perspectives: Optional[List[str]] = None,
    ) -> SummaryRecord:
        """
        Generate multi-perspective summaries.

        Parameters
        ----------
        scene_records : List[SceneRecord]
            Scene metadata from video analysis.
        emotion_records : List[EmotionRecord]
            Per-scene emotion data.
        subtitle_path : Optional[str]
            Path to an .srt file.
        output_dir : str
            Directory for generated HTML/JSON outputs.
        perspectives : Optional[List[str]]
            Subset of ['protagonist', 'antagonist', 'narrator'] to generate.

        Returns
        -------
        SummaryRecord
        """

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if all underlying dependencies are importable and functional."""


class IFusionEngine(abc.ABC):
    """
    Contract for fusing multimodal outputs into a final enriched summary.

    Concrete implementations wrap fusion/ module without modifying it.
    """

    @abc.abstractmethod
    def fuse(
        self,
        scene_records: List[SceneRecord],
        emotion_records: List[EmotionRecord],
        summary_record: SummaryRecord,
    ) -> FusedOutput:
        """
        Fuse video, emotion, and summary signals into a unified output.

        Parameters
        ----------
        scene_records : List[SceneRecord]
        emotion_records : List[EmotionRecord]
        summary_record : SummaryRecord

        Returns
        -------
        FusedOutput
        """

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if all underlying dependencies are importable and functional."""


class IEvaluator(abc.ABC):
    """
    Contract for computing evaluation metrics on generated summaries.

    Concrete implementations wrap evaluation/ module without modifying it.
    """

    @abc.abstractmethod
    def evaluate(
        self,
        fused_output: FusedOutput,
        reference_summary: Optional[str] = None,
        output_dir: str = "outputs/evaluation",
    ) -> EvaluationReport:
        """
        Compute all evaluation metrics for the generated output.

        Parameters
        ----------
        fused_output : FusedOutput
            The fused multimodal output to evaluate.
        reference_summary : Optional[str]
            Human reference summary for ROUGE/BLEU (optional).
        output_dir : str
            Directory to write evaluation artifacts.

        Returns
        -------
        EvaluationReport
        """

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if all underlying dependencies are importable and functional."""
