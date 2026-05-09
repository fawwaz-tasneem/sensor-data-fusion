"""
Tests for sensors.

For the Cartesian sensor we verify:
  - h(x) extracts position correctly for both 2D and 3D
  - H is the correct selector matrix
  - Detection probability behaves statistically
  - Measurements are unbiased (mean → truth as samples → infinity)
"""
import numpy as np

from sdf.core import StateLayout
from sdf.sensors import CartesianPositionSensor


class TestCartesianPositionSensor2D:
    def setup_method(self):
        self.layout = StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))
        self.sensor = CartesianPositionSensor(
            sensor_id="s1", dim=2, noise_std=1.0, detection_prob=1.0
        )

    def test_h_extracts_position(self):
        x = np.array([10.0, 5.0, 20.0, 7.0])
        z = self.sensor.h(x, self.layout)
        np.testing.assert_array_equal(z, [10.0, 20.0])

    def test_H_is_selector_matrix(self):
        x = np.zeros(4)
        H = self.sensor.H(x, self.layout)
        # Should pick out position indices (0, 2).
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=float)
        np.testing.assert_array_equal(H, expected)

    def test_measurement_is_unbiased(self):
        rng = np.random.default_rng(0)
        x_true = np.array([10.0, 0.0, 20.0, 0.0])
        # Average many measurements; mean should approach truth.
        samples = []
        for _ in range(10000):
            m = self.sensor.measure(x_true, self.layout, t=0.0, rng=rng)
            samples.append(m.value)
        mean = np.mean(samples, axis=0)
        # With 10000 samples and std=1, the std-error on the mean is ~0.01.
        np.testing.assert_allclose(mean, [10.0, 20.0], atol=0.1)


class TestDetectionProbability:
    def test_low_pd_yields_few_detections(self):
        layout = StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))
        sensor = CartesianPositionSensor(
            sensor_id="s1", dim=2, noise_std=1.0, detection_prob=0.3
        )
        rng = np.random.default_rng(0)
        x_true = np.zeros(4)
        n_trials = 5000
        n_detections = sum(
            sensor.measure(x_true, layout, t=0.0, rng=rng) is not None
            for _ in range(n_trials)
        )
        rate = n_detections / n_trials
        # With p=0.3 and 5000 trials, the std-error on the rate is ~0.006.
        assert abs(rate - 0.3) < 0.03


class TestCartesianPositionSensor3D:
    def test_h_in_3d(self):
        layout = StateLayout(
            dim=3, position_idx=(0, 2, 4), velocity_idx=(1, 3, 5)
        )
        sensor = CartesianPositionSensor(sensor_id="s1", dim=3)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        z = sensor.h(x, layout)
        np.testing.assert_array_equal(z, [1.0, 3.0, 5.0])
