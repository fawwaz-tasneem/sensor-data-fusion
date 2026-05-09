"""Tests for core data structures: StateLayout, StateDistribution, Measurement."""
import numpy as np
import pytest

from sdf.core import Measurement, StateDistribution, StateLayout


class TestStateLayout:
    def test_2d_layout_extracts_position(self):
        layout = StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))
        x = np.array([10.0, 1.0, 20.0, 2.0])
        np.testing.assert_array_equal(layout.position(x), [10.0, 20.0])
        np.testing.assert_array_equal(layout.velocity(x), [1.0, 2.0])

    def test_3d_layout_extracts_position(self):
        layout = StateLayout(
            dim=3, position_idx=(0, 2, 4), velocity_idx=(1, 3, 5)
        )
        x = np.array([10.0, 1.0, 20.0, 2.0, 30.0, 3.0])
        np.testing.assert_array_equal(layout.position(x), [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(layout.velocity(x), [1.0, 2.0, 3.0])

    def test_invalid_dim_raises(self):
        with pytest.raises(ValueError, match="dim must be 2 or 3"):
            StateLayout(dim=4, position_idx=(0, 1, 2, 3), velocity_idx=(4, 5, 6, 7))

    def test_mismatched_position_idx_raises(self):
        with pytest.raises(ValueError, match="position_idx must have"):
            StateLayout(dim=2, position_idx=(0,), velocity_idx=(1, 3))


class TestStateDistribution:
    def setup_method(self):
        self.layout = StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))

    def test_construction_validates_shapes(self):
        with pytest.raises(ValueError, match="mean must be 1D"):
            StateDistribution(
                mean=np.zeros((2, 2)),
                covariance=np.eye(4),
                timestamp=0.0,
                layout=self.layout,
            )

    def test_construction_validates_covariance_shape(self):
        with pytest.raises(ValueError, match="covariance shape"):
            StateDistribution(
                mean=np.zeros(4),
                covariance=np.eye(3),
                timestamp=0.0,
                layout=self.layout,
            )

    def test_copy_is_independent(self):
        s = StateDistribution(
            mean=np.array([1.0, 2.0, 3.0, 4.0]),
            covariance=np.eye(4),
            timestamp=0.0,
            layout=self.layout,
        )
        s_copy = s.copy()
        s_copy.mean[0] = 999.0
        # Original must be unchanged.
        assert s.mean[0] == 1.0


class TestMeasurement:
    def test_construction_validates_R_shape(self):
        with pytest.raises(ValueError, match="R shape"):
            Measurement(
                value=np.array([1.0, 2.0]),
                timestamp=0.0,
                sensor_id="s1",
                R=np.eye(3),
            )
