"""
Tests for DopplerBlindnessOcclusion.

Verify:
  1. Stationary target produces a low detection factor (in the notch).
  2. Fast-moving target produces a high detection factor.
  3. is_occluded is probabilistic when given an rng (different draws can
     differ).
  4. detection_factor stays within [pd_floor, 1.0].
  5. Behavior is symmetric in sign of range-rate.
"""
import numpy as np

from sdf.core.state import StateLayout
from sdf.sensors import DopplerBlindnessOcclusion


class TestDopplerBlindness:
    def setup_method(self):
        self.layout = StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))
        self.occ = DopplerBlindnessOcclusion(
            sensor_position=np.array([0.0, 0.0]),
            mdv=3.0,
            pd_floor=0.05,
        )

    def test_stationary_target_in_notch(self):
        # Target at (100, 100) with zero velocity: dot{r} = 0, in the notch.
        x = np.array([100.0, 0.0, 100.0, 0.0])
        f = self.occ.detection_factor(x, self.layout)
        # Expect close to pd_floor (since gaussian is at peak).
        assert f <= 0.1

    def test_fast_target_outside_notch(self):
        # Target at (100, 0) moving radially at 30 m/s: dot{r} = 30 >> mdv.
        x = np.array([100.0, 30.0, 0.0, 0.0])
        f = self.occ.detection_factor(x, self.layout)
        assert f > 0.95

    def test_factor_in_valid_range(self):
        # Sweep range-rates from -10 to +10 and check bounds.
        for vx in np.linspace(-10, 10, 41):
            x = np.array([100.0, vx, 0.0, 0.0])
            f = self.occ.detection_factor(x, self.layout)
            assert 0.05 <= f <= 1.0

    def test_symmetric_in_sign_of_range_rate(self):
        # At ±vx the detection factor should be the same.
        x_plus = np.array([100.0, 5.0, 0.0, 0.0])
        x_minus = np.array([100.0, -5.0, 0.0, 0.0])
        f_plus = self.occ.detection_factor(x_plus, self.layout)
        f_minus = self.occ.detection_factor(x_minus, self.layout)
        assert f_plus == f_minus

    def test_is_occluded_is_probabilistic(self):
        # Borderline case (factor ~ 0.5): different rng draws should
        # sometimes occlude and sometimes not.
        # To find a borderline case, sweep vx until factor ~ 0.5.
        layout = self.layout
        target_factor = 0.5
        best_vx = None
        best_diff = np.inf
        for vx in np.linspace(0, 10, 200):
            x = np.array([100.0, vx, 0.0, 0.0])
            f = self.occ.detection_factor(x, layout)
            if abs(f - target_factor) < best_diff:
                best_diff = abs(f - target_factor)
                best_vx = vx
        assert best_vx is not None
        x = np.array([100.0, best_vx, 0.0, 0.0])
        rng = np.random.default_rng(0)
        results = [self.occ.is_occluded(x, layout, rng) for _ in range(1000)]
        # Should see both True and False (not all the same).
        assert any(results) and not all(results)
