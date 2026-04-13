"""
calibration/__init__.py
========================
OCP-additive confidence calibration package.

Exposes the top-level EmotionCalibrator and diagnostic helpers.
"""
from calibration.confidence.calibration_layer import (
    EmotionCalibrator,
    diagnose_calibration,
    CalibrationDiagnostic,
)

__all__ = ["EmotionCalibrator", "diagnose_calibration", "CalibrationDiagnostic"]
