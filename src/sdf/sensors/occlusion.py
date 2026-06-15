"""
OcclusionModel: decides whether a target is visible to a sensor at a given
state. Pluggable so that terrain shadowing, GMTI MDV, and angular blind
zones can be composed independently and combined.

The contract is a single predicate, `is_occluded(target_state, layout, rng)`.
`rng` is part of the contract, not an afterthought: some occlusion models are
*probabilistic* (e.g. `DopplerBlindnessOcclusion` draws a Bernoulli trial with
a state-dependent success probability), so they need a random source. Purely
geometric models (e.g. `TunnelOcclusion`) ignore it. `rng` defaults to None so
callers that only use deterministic models don't have to thread one through;
probabilistic models fall back to a deterministic threshold when it's absent.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from sdf.core.state import StateLayout


class OcclusionModel(ABC):
    @abstractmethod
    def is_occluded(
        self,
        target_state: np.ndarray,
        layout: StateLayout,
        rng: Optional[np.random.Generator] = None,
    ) -> bool: ...


class CompositeOcclusion(OcclusionModel):
    """Logical OR of multiple occlusion models — occluded if any model says so."""

    def __init__(self, models: list[OcclusionModel]):
        self.models = models

    def is_occluded(
        self,
        target_state: np.ndarray,
        layout: StateLayout,
        rng: Optional[np.random.Generator] = None,
    ) -> bool:
        # Forward rng so probabilistic members (Doppler) keep sampling rather
        # than collapsing to their deterministic fallback.
        return any(m.is_occluded(target_state, layout, rng) for m in self.models)
