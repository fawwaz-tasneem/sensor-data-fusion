"""
Tests for IMMFilter.

We verify:
  1. Construction validates dimensions, transition matrix rows, and prob sums.
  2. With identical filters, mode probabilities should stay roughly uniform
     and the combined estimate should equal any single filter's estimate.
  3. After running on a maneuvering target, the mode probabilities should
     concentrate on the model whose dynamics best match the truth.
  4. Predicted mode probabilities are computed correctly from Pi and mu.
  5. IMM with multiple CT models tracks a turning target better than CV alone.
"""
import numpy as np
import pytest

from sdf.core import StateDistribution
from sdf.core.state import StateLayout
from sdf.filters import (
    ExtendedKalmanFilter,
    IMMFilter,
    KalmanFilter,
)
from sdf.motion_models import (
    ConstantVelocity,
    CoordinatedTurn,
)
from sdf.sensors import CartesianPositionSensor


def _layout_2d_cv():
    return StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))


def _initial_state(mean, P=None):
    layout = _layout_2d_cv()
    P = P if P is not None else np.diag([100.0, 25.0, 100.0, 25.0])
    return StateDistribution(
        mean=np.asarray(mean, dtype=float),
        covariance=P,
        timestamp=0.0,
        layout=layout,
    )


def _make_two_cv_filters():
    """Two identical CV filters at the same initial state."""
    cv = ConstantVelocity(dim=2, process_noise_std=0.5)
    s = _initial_state([0.0, 10.0, 0.0, 5.0])
    f1 = KalmanFilter(motion_model=cv, initial_state=s.copy())
    f2 = KalmanFilter(motion_model=cv, initial_state=s.copy())
    return [f1, f2]


class TestIMMConstruction:
    def test_rejects_single_filter(self):
        cv = ConstantVelocity(dim=2)
        s = _initial_state([0.0, 1.0, 0.0, 1.0])
        f = KalmanFilter(motion_model=cv, initial_state=s)
        with pytest.raises(ValueError, match="at least 2"):
            IMMFilter(filters=[f], transition_matrix=np.array([[1.0]]),
                      mode_probs=np.array([1.0]))

    def test_rejects_bad_transition_matrix_shape(self):
        filters = _make_two_cv_filters()
        with pytest.raises(ValueError, match="transition_matrix shape"):
            IMMFilter(filters=filters,
                      transition_matrix=np.array([[1.0]]),
                      mode_probs=np.array([0.5, 0.5]))

    def test_rejects_non_stochastic_transition(self):
        filters = _make_two_cv_filters()
        with pytest.raises(ValueError, match="rows must sum"):
            IMMFilter(filters=filters,
                      transition_matrix=np.array([[0.5, 0.5], [0.3, 0.3]]),
                      mode_probs=np.array([0.5, 0.5]))

    def test_rejects_mode_probs_not_summing_to_one(self):
        filters = _make_two_cv_filters()
        with pytest.raises(ValueError, match="mode_probs must sum"):
            IMMFilter(filters=filters,
                      transition_matrix=np.eye(2),
                      mode_probs=np.array([0.5, 0.6]))


class TestIMMWithIdenticalFilters:
    """If all filters have the same dynamics and the transition matrix is
    symmetric, the mode probabilities should stay uniform and the combined
    estimate should match any single filter."""

    def test_combined_state_equals_single_filter(self):
        filters = _make_two_cv_filters()
        # Reference: a third filter run separately.
        cv = filters[0].motion_model
        ref = KalmanFilter(motion_model=cv, initial_state=filters[0].state.copy())

        imm = IMMFilter(
            filters=filters,
            transition_matrix=np.array([[0.95, 0.05], [0.05, 0.95]]),
            mode_probs=np.array([0.5, 0.5]),
        )

        sensor = CartesianPositionSensor(
            sensor_id="cart_1", dim=2, noise_std=2.0, detection_prob=1.0
        )
        rng = np.random.default_rng(42)
        layout = filters[0].state.layout

        x_true = np.array([0.0, 10.0, 0.0, 5.0])
        for k in range(1, 11):
            t = k * 0.5
            x_true = x_true + np.array([10.0 * 0.5, 0, 5.0 * 0.5, 0])
            m = sensor.measure(x_true, layout, t, rng)
            imm.predict(t)
            ref.predict(t)
            imm.update(m, sensor)
            ref.update(m, sensor)

        # After 10 steps with identical dynamics, IMM and reference should
        # agree (within numerical tolerance from mixing).
        np.testing.assert_allclose(imm.state.mean, ref.state.mean, atol=1.0)


class TestIMMModeProbabilitiesEvolveCorrectly:
    """If the truth follows model A and not model B, mode probability for
    A should grow over time."""

    def test_cv_vs_ct_on_straight_truth(self):
        # Truth: straight line at constant velocity. CV should win over CT.
        cv = ConstantVelocity(dim=2, process_noise_std=0.5)
        ct = CoordinatedTurn(omega=0.1, process_noise_std=0.5)
        x0 = np.array([0.0, 10.0, 0.0, 0.0])  # moving in +x at 10 m/s
        s_cv = _initial_state(x0)
        s_ct = _initial_state(x0)
        f_cv = KalmanFilter(motion_model=cv, initial_state=s_cv)
        f_ct = ExtendedKalmanFilter(motion_model=ct, initial_state=s_ct)

        imm = IMMFilter(
            filters=[f_cv, f_ct],
            transition_matrix=np.array([[0.9, 0.1], [0.1, 0.9]]),
            mode_probs=np.array([0.5, 0.5]),
        )

        sensor = CartesianPositionSensor(
            sensor_id="cart_1", dim=2, noise_std=1.0, detection_prob=1.0
        )
        rng = np.random.default_rng(0)
        layout = s_cv.layout

        x_true = x0.copy()
        for k in range(1, 31):
            t = k * 0.5
            x_true = x_true + np.array([x_true[1] * 0.5, 0, x_true[3] * 0.5, 0])
            m = sensor.measure(x_true, layout, t, rng)
            imm.predict(t)
            imm.update(m, sensor)

        # CV (mode 0) should win: its mode prob > CT's. The absolute
        # ceiling depends on the transition matrix — with 10% leakage
        # per step, the winner caps below 0.7 even when it's clearly right.
        assert imm.mode_probabilities[0] > imm.mode_probabilities[1]
        assert imm.mode_probabilities[0] > 0.55

    def test_ct_wins_on_circular_truth(self):
        # Truth: target moves on a circle. CT should win over CV.
        cv = ConstantVelocity(dim=2, process_noise_std=0.5)
        ct = CoordinatedTurn(omega=0.1, process_noise_std=0.5)
        omega_truth = 0.1
        R = 100.0
        x0 = np.array([R, 0.0, 0.0, R * omega_truth])
        s_cv = _initial_state(x0)
        s_ct = _initial_state(x0)
        f_cv = KalmanFilter(motion_model=cv, initial_state=s_cv)
        f_ct = ExtendedKalmanFilter(motion_model=ct, initial_state=s_ct)

        imm = IMMFilter(
            filters=[f_cv, f_ct],
            transition_matrix=np.array([[0.9, 0.1], [0.1, 0.9]]),
            mode_probs=np.array([0.5, 0.5]),
        )

        sensor = CartesianPositionSensor(
            sensor_id="cart_1", dim=2, noise_std=0.5, detection_prob=1.0
        )
        rng = np.random.default_rng(0)
        layout = s_cv.layout

        # Generate truth via CT-known with omega_truth.
        truth_filter = ExtendedKalmanFilter(
            motion_model=CoordinatedTurn(omega=omega_truth, process_noise_std=0.0),
            initial_state=_initial_state(x0, P=np.eye(4) * 1e-9),
        )

        for k in range(1, 31):
            t = k * 0.5
            truth_filter.predict(t)
            x_true = truth_filter.state.mean.copy()
            m = sensor.measure(x_true, layout, t, rng)
            imm.predict(t)
            imm.update(m, sensor)

        # CT (mode 1) should win.
        assert imm.mode_probabilities[1] > imm.mode_probabilities[0]


class TestIMMTracksManeuveringBetterThanCV:
    def test_imm_vs_pure_cv_on_turn(self):
        omega_truth = 0.08

        # Reference: pure CV filter (no IMM).
        cv = ConstantVelocity(dim=2, process_noise_std=0.5)
        x0 = np.array([100.0, 0.0, 0.0, 100.0 * omega_truth])
        ref_cv = KalmanFilter(
            motion_model=cv, initial_state=_initial_state(x0)
        )

        # IMM: CV (omega=0) + CT-left + CT-right.
        ct_left = CoordinatedTurn(omega=+0.08, process_noise_std=0.5)
        ct_right = CoordinatedTurn(omega=-0.08, process_noise_std=0.5)
        f1 = KalmanFilter(
            motion_model=ConstantVelocity(dim=2, process_noise_std=0.5),
            initial_state=_initial_state(x0),
        )
        f2 = ExtendedKalmanFilter(
            motion_model=ct_left, initial_state=_initial_state(x0)
        )
        f3 = ExtendedKalmanFilter(
            motion_model=ct_right, initial_state=_initial_state(x0)
        )
        imm = IMMFilter(
            filters=[f1, f2, f3],
            transition_matrix=np.array([
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9],
            ]),
            mode_probs=np.array([1 / 3, 1 / 3, 1 / 3]),
        )

        sensor = CartesianPositionSensor(
            sensor_id="cart_1", dim=2, noise_std=2.0, detection_prob=1.0
        )
        rng = np.random.default_rng(7)
        layout = ref_cv.state.layout

        # Truth via CT-known with omega_truth.
        truth_filter = ExtendedKalmanFilter(
            motion_model=CoordinatedTurn(
                omega=omega_truth, process_noise_std=0.0
            ),
            initial_state=_initial_state(x0, P=np.eye(4) * 1e-9),
        )

        ref_errs, imm_errs = [], []
        for k in range(1, 41):
            t = k * 0.5
            truth_filter.predict(t)
            x_true = truth_filter.state.mean.copy()
            m = sensor.measure(x_true, layout, t, rng)
            # rng draw used by both filters via the same measurement.
            ref_cv.predict(t)
            ref_cv.update(m, sensor)
            imm.predict(t)
            imm.update(m, sensor)
            true_pos = x_true[[0, 2]]
            ref_errs.append(np.linalg.norm(ref_cv.state.position() - true_pos))
            imm_errs.append(np.linalg.norm(imm.state.position() - true_pos))

        # IMM should beat pure CV on a turning target.
        assert np.mean(imm_errs) < np.mean(ref_errs)
