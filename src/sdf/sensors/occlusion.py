"""
OcclusionModel: decides whether a target is visible to a sensor at a given
state. Pluggable so that terrain shadowing, GMTI MDV, and angular blind
zones can be composed independently and combined.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from sdf.core.state import StateLayout


class OcclusionModel(ABC):
    @abstractmethod
    def is_occluded(self, target_state: np.ndarray, layout: StateLayout) -> bool: ...


class CompositeOcclusion(OcclusionModel):
    """Logical OR of multiple occlusion models — occluded if any model says so."""

    def __init__(self, models: list[OcclusionModel]):
        self.models = models

    def is_occluded(self, target_state: np.ndarray, layout: StateLayout) -> bool:
        return any(m.is_occluded(target_state, layout) for m in self.models)
