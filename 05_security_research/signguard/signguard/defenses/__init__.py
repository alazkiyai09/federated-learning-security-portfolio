"""Baseline defense implementations for SignGuard."""

from signguard.defenses.krum import KrumDefense
from signguard.defenses.trimmed_mean import TrimmedMeanDefense
from signguard.defenses.foolsgold import FoolsGoldDefense
from signguard.defenses.bulyan import BulyanDefense

__all__ = [
    "KrumDefense",
    "TrimmedMeanDefense",
    "FoolsGoldDefense",
    "BulyanDefense",
]
