"""Anomaly detection modules for SignGuard."""

from signguard.detection.base import AnomalyDetector
from signguard.detection.magnitude_detector import L2NormDetector
from signguard.detection.direction_detector import CosineSimilarityDetector
from signguard.detection.score_detector import LossDeviationDetector
from signguard.detection.ensemble import EnsembleDetector

__all__ = [
    "AnomalyDetector",
    "EnsembleDetector",
    "L2NormDetector",
    "CosineSimilarityDetector",
    "LossDeviationDetector",
]
