"""
research_layers/adaptive_fusion/__init__.py
=============================================
Stub sub-package placeholder for adaptive fusion research layer.

Adaptive fusion implementation lives in:
  fusion_plus/adaptive_fusion.py  -> AdaptiveModalityFusion, adaptive_fuse_numpy

This directory is reserved for future ablation studies comparing
different fusion strategies (static, learned, attention-based, etc.)
within the research_layers experimental framework.
"""
# Re-export from canonical OCP extension location
from fusion_plus.adaptive_fusion import adaptive_fuse_numpy

__all__ = ["adaptive_fuse_numpy"]
