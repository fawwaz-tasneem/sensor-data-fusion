"""
Tests for RoadAidedExtendedKalmanFilter.

Cornerstone tests:
  1. The road update pulls a position off-road toward the road.
  2. After many road updates with no other measurements, the cross-track
     uncertainty shrinks but the along-track uncertainty grows freely.
  3. Combined with sensor updates, road aiding reduces position error
     compared to a plain EKF without the road.
  4. The filter associates with the correct segment in an L-shaped road.
"""
import numpy as np
import pytest

from sdf.core import StateDistribution
from sdf.filters import (
    ExtendedKalmanFilter,
    RoadAidedExtendedKalmanFilter,
)
from sdf.motion_models import ConstantVelocity
from sdf.scenarios import PolygonalRoadMap
from sdf.sensors import RadarSensor


def _straight_2d_road():
    nodes = np.array([[0.0, 0.0], [1000.0, 0.0]])
    return PolygonalRoadMap(nodes, sigma_nodes=2.0)


def _initial_state(layout, mean, P=None):
    P = P if P is not None else np.diag([100.0, 25.0, 100.0, 25.0])
    return StateDistribution(
        mean=np.asarray(mean, dtype=float),
        covariance=P,
        timestamp=0.0,
        layout=layout,
    )


class TestRoadUpdate:
    def setup_method(self):
        self.cv = ConstantVelocity(dim=2, process_noise_std=0.5)
        self.road = _straight_2d_road()

    def test_pulls_off_road_toward_road(self):
        # State at (500, 50) moving at (10, 0). Road is the x-axis.
        # After one road update, y should drop substantially toward 0.
        init = _initial_state(self.cv.layout, [500.0, 10.0, 50.0, 0.0])
        f = RoadAidedExtendedKalmanFilter(
            motion_model=self.cv, initial_state=init, road_map=self.road
        )
        y_before = f.state.position()[1]
        f.update_with_road()
        y_after = f.state.position()[1]
        assert abs(y_after) < abs(y_before)

    def test_along_track_unaffected_by_single_update(self):
        # x-position should be essentially unchanged by a road update on a
        # straight road, because the road only constrains cross-track.
        init = _initial_state(self.cv.layout, [500.0, 10.0, 50.0, 0.0])
        f = RoadAidedExtendedKalmanFilter(
            motion_model=self.cv, initial_state=init, road_map=self.road
        )
        x_before = f.state.position()[0]
        f.update_with_road()
        x_after = f.state.position()[0]
        # Numerically should be exactly unchanged for a straight road
        # because the cross-track normal is perpendicular to x.
        assert abs(x_after - x_before) < 1e-8

    def test_cross_track_covariance_shrinks(self):
        # The y-y covariance entry should shrink after a road update.
        init = _initial_state(self.cv.layout, [500.0, 10.0, 50.0, 0.0])
        f = RoadAidedExtendedKalmanFilter(
            motion_model=self.cv, initial_state=init, road_map=self.road
        )
        # y is index 2 in the state vector.
        var_y_before = f.state.covariance[2, 2]
        f.update_with_road()
        var_y_after = f.state.covariance[2, 2]
        assert var_y_after < var_y_before


class TestRoadVsNoRoad:
    """Road aiding should reduce error vs. plain EKF on the same scenario."""

    def test_road_aided_beats_plain_ekf(self):
        cv = ConstantVelocity(dim=2, process_noise_std=0.3)
        road = _straight_2d_road()
        sensor = RadarSensor(
            sensor_id="r1",
            position=np.array([0.0, -500.0]),
            range_std=20.0,
            bearing_std=1e-2,
            detection_prob=1.0,
        )

        # True trajectory: along x-axis at 10 m/s.
        truth_init = np.array([0.0, 10.0, 0.0, 0.0])
        # Both filters start at the SAME prior, deliberately offset off-road.
        prior = _initial_state(
            cv.layout,
            [50.0, 8.0, 30.0, 0.0],  # 30 m off-road
            P=np.diag([400.0, 16.0, 400.0, 16.0]),
        )

        plain = ExtendedKalmanFilter(motion_model=cv, initial_state=prior.copy())
        aided = RoadAidedExtendedKalmanFilter(
            motion_model=cv, initial_state=prior.copy(), road_map=road
        )

        rng = np.random.default_rng(0)
        plain_errs, aided_errs = [], []
        for k in range(1, 101):
            t = k * 0.5
            x_true = np.array(
                [truth_init[0] + truth_init[1] * t, truth_init[1], 0.0, 0.0]
            )
            m = sensor.measure(x_true, cv.layout, t, rng)
            plain.predict(t)
            plain.update(m, sensor)
            aided.predict(t)
            aided.update(m, sensor)
            aided.update_with_road()
            plain_errs.append(
                np.linalg.norm(plain.state.position() - x_true[[0, 2]])
            )
            aided_errs.append(
                np.linalg.norm(aided.state.position() - x_true[[0, 2]])
            )

        # Road-aided should be at least as good on average.
        assert np.mean(aided_errs) < np.mean(plain_errs)


class TestSegmentAssociation:
    def test_l_shaped_road_picks_correct_segment(self):
        # L-shaped road: along x then along y.
        nodes = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        road = PolygonalRoadMap(nodes, sigma_nodes=1.0)
        cv = ConstantVelocity(dim=2, process_noise_std=0.1)
        # State near second segment at (12, 5).
        init = _initial_state(cv.layout, [12.0, 0.0, 5.0, 1.0])
        f = RoadAidedExtendedKalmanFilter(
            motion_model=cv, initial_state=init, road_map=road
        )
        f.update_with_road()
        assert f.last_segment_idx == 1


def test_road_map_dim_must_match_layout_dim():
    cv = ConstantVelocity(dim=2)
    init = _initial_state(cv.layout, [0.0, 1.0, 0.0, 1.0])
    nodes_3d = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    road_3d = PolygonalRoadMap(nodes_3d)
    with pytest.raises(ValueError, match="!="):
        RoadAidedExtendedKalmanFilter(
            motion_model=cv, initial_state=init, road_map=road_3d
        )


class TestRoadCrossTrackUpdateReusable:
    """
    road_cross_track_update is the standalone engine behind road aiding;
    it must constrain ANY filter's state to the road, not just the
    RoadAidedExtendedKalmanFilter. This is what lets the dashboard fuse the
    road map into a plain EKF.
    """

    def test_pulls_plain_ekf_estimate_onto_road(self):
        import numpy as np
        from sdf.core.state import StateDistribution, StateLayout
        from sdf.filters import ExtendedKalmanFilter, road_cross_track_update
        from sdf.motion_models import ConstantVelocity
        from sdf.scenarios import PolygonalRoadMap

        # Straight road along +x at y=0, z=0.
        nodes = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0],
                          [2000.0, 0.0, 0.0]])
        road = PolygonalRoadMap(nodes=nodes, sigma_nodes=1.0)
        cv = ConstantVelocity(dim=3, process_noise_std=1.0)

        # Start the estimate well OFF the road (200 m cross-track in y and z).
        mean = np.array([500.0, 10.0, 200.0, 0.0, 200.0, 0.0])
        cov = np.diag([100.0, 25, 1e4, 25, 1e4, 25])  # loose cross-track prior
        state = StateDistribution(mean, cov, 0.0, cv.layout)

        before = np.linalg.norm(road.closest_segment(cv.layout.position(state.mean))[2:])
        new_state, seg, foot = road_cross_track_update(state, road)
        _, _, after = road.closest_segment(cv.layout.position(new_state.mean))

        assert before > 100.0
        assert after < 5.0  # pulled essentially onto the road
