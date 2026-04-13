"""
integration/__init__.py
========================
Top-level integration package.

Exposes the two main entry points consumers need:
  - ServiceRegistry  (build and look up adapters)
  - PipelineOrchestrator  (run the full pipeline)
"""
from .registry.service_registry import ServiceRegistry
from .pipeline.orchestrator import PipelineOrchestrator

__all__ = ["ServiceRegistry", "PipelineOrchestrator"]
