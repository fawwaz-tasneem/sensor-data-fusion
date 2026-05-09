"""
A Track is the time-history of a filter's estimates for one target.

It exists primarily so that smoothers can consume
forward-pass output, and so that visualization/metrics can replay an
estimation run without re-running the filter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sdf.core.state import StateDistribution


@dataclass
class Track:
    track_id: str
    history: list[StateDistribution] = field(default_factory=list)

    def append(self, state: StateDistribution) -> None:
        self.history.append(state)

    def __len__(self) -> int:
        return len(self.history)

    @property
    def latest(self) -> StateDistribution:
        return self.history[-1]

    def positions(self) -> np.ndarray:
        """Stacked position estimates, shape (T, dim)."""
        return np.array([s.position() for s in self.history])

    def timestamps(self) -> np.ndarray:
        return np.array([s.timestamp for s in self.history])
