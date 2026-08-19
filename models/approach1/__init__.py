"""Configurable staged model for spatial polyadic partner assignment."""

from .config import DataPaths, DegreeSummary, FeatureFlags, ModelConfig
from .model import PolyadicAssignmentModel, PredictionResult

__all__ = [
    "DataPaths",
    "DegreeSummary",
    "FeatureFlags",
    "ModelConfig",
    "PolyadicAssignmentModel",
    "PredictionResult",
]
