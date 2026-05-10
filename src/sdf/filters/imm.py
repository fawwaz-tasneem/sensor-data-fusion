"""
Interacting Multiple Models (IMM) filter.

The IMM runs r member filters in parallel, each operating under a different
motion model. At every step it maintains a probability mu_i that the
target is in mode i. The filter output is a Gaussian-mixture-collapsed
state from all members, weighted by mu_i.

Algorithm (Bar-Shalom, Estimation with Applications to Tracking, Sec. 11.6):

  Pre-step (mixing):
    1. Predicted mode probabilities:
         mu_{i|j}(k|k-1) = (Pi_{ij} mu_i(k-1)) / c_j
       where c_j = sum_i Pi_{ij} mu_i(k-1).
    2. Mixed initial conditions for filter j:
         x_{0j} = sum_i mu_{i|j}(k|k-1) x_i(k-1)
         P_{0j} = sum_i mu_{i|j}(k|k-1) [P_i + (x_i - x_{0j})(x_i - x_{0j})^T]

  Filtering:
    Each filter predicts and updates with its own dynamics and the same
    measurement, computing its own likelihood Lambda_j of the measurement.

  Mode update:
    mu_j(k) = (Lambda_j c_j) / sum_l (Lambda_l c_l)

  Output combination:
    x_out = sum_j mu_j(k) x_j(k)
    P_out = sum_j mu_j(k) [P_j + (x_j - x_out)(x_j - x_out)^T]

All member filters MUST share the same state layout (and therefore the
same state dimension). To mix models that naturally have different state
dimensions (e.g., CV vs. CT-unknown), the caller must augment the lower-
dim model into the higher-dim layout — typically by carrying a constant
"unused" component with high process noise so it doesn't influence the
estimate.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sdf.core.measurement import Measurement
from sdf.core.state import StateDistribution
from sdf.filters.base import Filter
from sdf.sensors.base import Sensor


class IMMFilter(Filter):
    """Interacting Multiple Models filter."""

    def __init__(
        self,
        filters: list[Filter],
        transition_matrix: np.ndarray,
        mode_probs: np.ndarray,
    ):
        """
        Parameters
        ----------
        filters : list of Filter
            Member filters, one per motion mode. All must share the same
            initial state layout.
        transition_matrix : (r, r) array
            Pi[i, j] = P(mode_{k+1} = j | mode_k = i). Rows must sum to 1.
        mode_probs : (r,) array
            Initial mode probabilities. Must sum to 1.
        """
        if len(filters) < 2:
            raise ValueError("IMM requires at least 2 filters")
        r = len(filters)
        Pi = np.asarray(transition_matrix, dtype=float)
        mu = np.asarray(mode_probs, dtype=float)
        if Pi.shape != (r, r):
            raise ValueError(f"transition_matrix shape {Pi.shape} != ({r},{r})")
        if mu.shape != (r,):
            raise ValueError(f"mode_probs shape {mu.shape} != ({r},)")
        if not np.allclose(Pi.sum(axis=1), 1.0):
            raise ValueError("transition_matrix rows must sum to 1")
        if not np.isclose(mu.sum(), 1.0):
            raise ValueError("mode_probs must sum to 1")

        # All filters must share the same state layout.
        ref_layout = filters[0].state.layout
        for i, f in enumerate(filters[1:], 1):
            if f.state.layout is not ref_layout and (
                f.state.layout.dim != ref_layout.dim
                or f.state.layout.position_idx != ref_layout.position_idx
                or f.state.layout.velocity_idx != ref_layout.velocity_idx
            ):
                raise ValueError(
                    f"filter {i}'s layout differs from filter 0's layout"
                )
            if f.state.mean.shape != filters[0].state.mean.shape:
                raise ValueError(
                    f"filter {i}'s state dim {f.state.mean.shape[0]} "
                    f"!= filter 0's {filters[0].state.mean.shape[0]}"
                )

        self.filters = filters
        self.Pi = Pi
        self.mu = mu
        # The "current" combined state is exposed via self.state for the
        # outside world (Filter base class contract).
        self.motion_model = filters[0].motion_model  # nominal; not actually used
        self._update_combined_state()

    # ----- Public Filter interface -------------------------------------

    def predict(self, t: float) -> StateDistribution:
        """
        IMM predict step:
          - Mix initial conditions across filters
          - Each filter independently predicts to t
          - Re-combine for the public self.state

        We do NOT update mode probabilities here; that happens after a
        measurement is processed (in update()), because mode-probability
        updates need the measurement likelihoods.
        """
        # 1. Compute mixing weights w_{i|j} = Pi_{ij} mu_i / c_j with c_j = sum_i Pi_{ij} mu_i.
        # Layout: w[i, j] = mu_{i|j}.
        c = self.Pi.T @ self.mu  # shape (r,)
        # Avoid division by zero in case some c_j is ~0 (a "dead" mode).
        c_safe = np.where(c > 1e-12, c, 1.0)
        # w_{i|j} = Pi[i,j] * mu[i] / c[j]
        W = self.Pi * self.mu[:, None] / c_safe[None, :]

        # 2. Mixed initial conditions for each filter j.
        # We need to set each filter's state to the mixed distribution
        # (mean and covariance) before letting it predict.
        n = self.filters[0].state.mean.shape[0]
        means_prev = np.array([f.state.mean for f in self.filters])  # (r, n)
        covs_prev = np.array([f.state.covariance for f in self.filters])  # (r, n, n)

        for j, fj in enumerate(self.filters):
            # Mixed mean.
            x0_j = (W[:, j][:, None] * means_prev).sum(axis=0)
            # Mixed covariance.
            P0_j = np.zeros((n, n))
            for i in range(len(self.filters)):
                diff = (means_prev[i] - x0_j)[:, None]
                P0_j += W[i, j] * (covs_prev[i] + diff @ diff.T)
            # Replace fj's state with the mixed distribution. Timestamp
            # stays at the previous step (the filter will advance to t
            # when predict() is called).
            fj.state = StateDistribution(
                mean=x0_j,
                covariance=P0_j,
                timestamp=fj.state.timestamp,
                layout=fj.state.layout,
            )

        # 3. Each filter predicts independently to t.
        for fj in self.filters:
            fj.predict(t)

        # 4. The c vector (predicted mode probabilities before any
        # measurement) is needed by update() to combine mode probabilities;
        # cache it here.
        self._mixing_c = c

        self._update_combined_state()
        return self.state

    def update(self, measurement: Measurement, sensor: Sensor) -> StateDistribution:
        """
        IMM update step:
          - Each filter independently updates with the measurement,
            recording its likelihood.
          - Mode probabilities are updated via Bayes.
          - Combined state is recomputed.
        """
        likelihoods = np.zeros(len(self.filters))
        for j, fj in enumerate(self.filters):
            # Compute the measurement likelihood under filter j BEFORE update.
            # The likelihood is N(z; z_pred, S) at the predicted state.
            x_pred = fj.state.mean
            P_pred = fj.state.covariance
            H = sensor.H(x_pred, fj.state.layout)
            z_pred = sensor.h(x_pred, fj.state.layout)
            y = sensor.innovation(measurement.value, z_pred)
            S = H @ P_pred @ H.T + measurement.R

            # Gaussian likelihood: (2 pi)^{-m/2} |S|^{-1/2} exp(-1/2 y^T S^{-1} y)
            try:
                # Use slogdet for numerical stability.
                sign, logdet = np.linalg.slogdet(S)
                if sign <= 0:
                    likelihoods[j] = 1e-300
                else:
                    # S y_solve = y
                    y_solve = np.linalg.solve(S, y)
                    quad = float(y @ y_solve)
                    m = y.shape[0]
                    log_lik = -0.5 * (m * np.log(2 * np.pi) + logdet + quad)
                    likelihoods[j] = float(np.exp(log_lik))
            except np.linalg.LinAlgError:
                likelihoods[j] = 1e-300

            # Now run the actual filter update.
            fj.update(measurement, sensor)

        # Update mode probabilities: mu_j ∝ Lambda_j * c_j.
        c = getattr(self, "_mixing_c", self.Pi.T @ self.mu)
        unnormalized = likelihoods * c
        total = unnormalized.sum()
        if total > 1e-300:
            self.mu = unnormalized / total
        # If total likelihood is essentially zero (a hugely unlikely
        # measurement under all modes), keep mode probs unchanged rather
        # than divide by ~0.

        self._update_combined_state()
        return self.state

    # ----- Helpers -----------------------------------------------------

    def _update_combined_state(self) -> None:
        """Combine the current per-filter posteriors into self.state."""
        n = self.filters[0].state.mean.shape[0]
        means = np.array([f.state.mean for f in self.filters])
        covs = np.array([f.state.covariance for f in self.filters])
        x_combined = (self.mu[:, None] * means).sum(axis=0)
        P_combined = np.zeros((n, n))
        for j in range(len(self.filters)):
            diff = (means[j] - x_combined)[:, None]
            P_combined += self.mu[j] * (covs[j] + diff @ diff.T)
        # Use the first filter's timestamp/layout as canonical.
        self.state = StateDistribution(
            mean=x_combined,
            covariance=P_combined,
            timestamp=self.filters[0].state.timestamp,
            layout=self.filters[0].state.layout,
        )

    @property
    def mode_probabilities(self) -> np.ndarray:
        return self.mu.copy()
