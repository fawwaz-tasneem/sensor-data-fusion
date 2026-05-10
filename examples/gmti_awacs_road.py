"""
Moving GMTI radar (AWACS) + road map.

A ground vehicle moves along a 2D road, stops in the middle (entering
the GMTI clutter notch via Doppler blindness), and resumes. The vehicle
is tracked by a single moving GMTI sensor on an AWACS-like racetrack
flight pattern, low and slow, close to the road.

We compare two filter configurations:
  A. Plain EKF + GMTI only
  B. Road-aided EKF + GMTI

The road constraint helps significantly during the vehicle's stop, when
the GMTI loses most detections to Doppler blindness and the filter must
coast on the road constraint and motion model alone.

Outputs:
  results/gmti_awacs_road.png - top-down view + per-step error +
                                  GMTI detection events
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sdf.core.state import StateDistribution
from sdf.filters import ExtendedKalmanFilter, RoadAidedExtendedKalmanFilter
from sdf.motion_models import ConstantVelocity
from sdf.scenarios import PolygonalRoadMap, RacetrackFlight
from sdf.sensors import DopplerBlindnessOcclusion, GMTIRadarSensor


# ---- Scenario parameters ----------------------------------------------

V_MS = 12.0  # vehicle cruising speed (m/s, ~43 km/h)
T_TOTAL = 240.0  # total simulation time
T_STOP_START = 80.0
T_STOP_END = 140.0
DT = 1.0  # 1 Hz sensor scan rate
MDV = 2.0  # GMTI Minimum Detectable Velocity, m/s


def true_state(t: float) -> np.ndarray:
    """Vehicle that moves along +x, stops, then resumes."""
    if t <= T_STOP_START:
        x = V_MS * t
        vx = V_MS
    elif t <= T_STOP_END:
        x = V_MS * T_STOP_START
        vx = 0.0
    else:
        x = V_MS * T_STOP_START + V_MS * (t - T_STOP_END)
        vx = V_MS
    # State [x, vx, y, vy] (2D CV layout).
    return np.array([x, vx, 0.0, 0.0])


def build_road_map() -> PolygonalRoadMap:
    """Straight road along the x-axis."""
    nodes = np.array([
        [-200.0, 0.0],
        [1000.0, 0.0],
        [2500.0, 0.0],
        [4000.0, 0.0],
    ])
    return PolygonalRoadMap(nodes, sigma_nodes=3.0)


def main() -> None:
    cv = ConstantVelocity(dim=2, process_noise_std=1.5)
    layout = cv.layout
    road = build_road_map()

    # AWACS racetrack platform: low and slow, perpendicular offset of ~200 m
    # from the road. In this 2D scenario "altitude" is the y-coordinate
    # offset; 200 m here is the stand-in for a low-flying surveillance
    # platform that stays close to the action. Speed at the minimum
    # plausible cruising speed for a fixed-wing surveillance aircraft.
    awacs = RacetrackFlight(
        center=np.array([1500.0, 200.0]),
        leg_length=3000.0,
        radius=400.0,
        speed=60.0,
    )

    # GMTI on AWACS with Doppler blindness.
    blindness = DopplerBlindnessOcclusion(
        sensor_position=awacs.position_at(0.0),
        mdv=MDV,
    )
    gmti = GMTIRadarSensor(
        sensor_id="gmti_awacs",
        position=awacs.position_at(0.0),
        range_std=20.0,
        bearing_std=3e-3,
        range_rate_std=0.5,
        detection_prob=1.0,
        occlusion_model=blindness,
        platform=awacs,
    )

    # Common initial estimate.
    truth_at_zero = true_state(0.0)
    init = StateDistribution(
        mean=truth_at_zero.copy(),
        covariance=np.diag([100.0, 16.0, 100.0, 16.0]),
        timestamp=0.0,
        layout=layout,
    )

    plain_ekf = ExtendedKalmanFilter(motion_model=cv, initial_state=init.copy())
    aided_ekf = RoadAidedExtendedKalmanFilter(
        motion_model=cv, initial_state=init.copy(), road_map=road
    )

    rng = np.random.default_rng(3)

    times = np.arange(0.0, T_TOTAL + DT, DT)
    truths, plain_track, aided_track = [], [], []
    gmti_detections = []
    awacs_track = []

    for t in times:
        x_true = true_state(t)
        truths.append(x_true)
        awacs_track.append(awacs.position_at(t))

        if t == 0.0:
            plain_track.append(plain_ekf.state.mean.copy())
            aided_track.append(aided_ekf.state.mean.copy())
            gmti_detections.append(False)
            continue

        plain_ekf.predict(t)
        aided_ekf.predict(t)

        # Apply the moving GMTI (sets sensor pose to platform's t).
        m_gmti = gmti.measure(x_true, layout, t, rng)
        if m_gmti is not None:
            plain_ekf.update(m_gmti, gmti)
            aided_ekf.update(m_gmti, gmti)
            gmti_detections.append(True)
        else:
            gmti_detections.append(False)

        # Road constraint on the aided filter only.
        aided_ekf.update_with_road()

        plain_track.append(plain_ekf.state.mean.copy())
        aided_track.append(aided_ekf.state.mean.copy())

    truths_a = np.array(truths)
    plain_a = np.array(plain_track)
    aided_a = np.array(aided_track)
    awacs_a = np.array(awacs_track)

    truth_pos = truths_a[:, [0, 2]]
    plain_pos = plain_a[:, [0, 2]]
    aided_pos = aided_a[:, [0, 2]]
    plain_err = np.linalg.norm(plain_pos - truth_pos, axis=1)
    aided_err = np.linalg.norm(aided_pos - truth_pos, axis=1)

    n_gmti_total = sum(1 for d in gmti_detections[1:] if d)
    n_gmti_during_stop = sum(
        1 for t, d in zip(times[1:], gmti_detections[1:])
        if T_STOP_START < t <= T_STOP_END and d
    )
    n_stop_steps = sum(
        1 for t in times[1:] if T_STOP_START < t <= T_STOP_END
    )

    print(f"Plain EKF mean position error: {plain_err[1:].mean():.2f} m  "
          f"(max {plain_err[1:].max():.2f})")
    print(f"Road-aided EKF mean position error: {aided_err[1:].mean():.2f} m  "
          f"(max {aided_err[1:].max():.2f})")
    print(f"GMTI detections overall: {n_gmti_total} / {len(times)-1}")
    print(f"GMTI detections during vehicle stop: "
          f"{n_gmti_during_stop} / {n_stop_steps}  "
          f"(Doppler blindness suppresses these)")

    # Plot.
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gmti_awacs_road.png")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1])

    # Top-down view: vehicle, road, AWACS path, estimates.
    ax = fig.add_subplot(gs[0, :])
    ax.plot(truth_pos[:, 0], truth_pos[:, 1], color="green", linewidth=1.5,
            label="Vehicle (truth)")
    ax.plot(road.nodes[:, 0], road.nodes[:, 1], "ko-", markersize=3,
            linewidth=0.7, label="Road map")
    ax.plot(awacs_a[:, 0], awacs_a[:, 1], color="orange", linewidth=0.8,
            label="AWACS racetrack", alpha=0.7)
    ax.plot(plain_pos[:, 0], plain_pos[:, 1], color="red", linewidth=1.0,
            label="Plain EKF", alpha=0.8)
    ax.plot(aided_pos[:, 0], aided_pos[:, 1], color="blue", linewidth=1.0,
            label="Road-aided EKF", alpha=0.8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("GMTI-on-AWACS + road map")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    # Error over time.
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(times, plain_err, color="red", linewidth=1.0,
            label="Plain EKF", alpha=0.85)
    ax.plot(times, aided_err, color="blue", linewidth=1.0,
            label="Road-aided EKF", alpha=0.85)
    ax.axvspan(T_STOP_START, T_STOP_END, color="gray", alpha=0.2,
               label="Vehicle stopped")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Position error [m]")
    ax.set_title("Position error over time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # GMTI detection events.
    ax = fig.add_subplot(gs[1, 1])
    detection_array = np.array([1 if d else 0 for d in gmti_detections])
    ax.fill_between(times, 0, detection_array, step="post", color="green",
                    alpha=0.6, label="GMTI detection")
    ax.axvspan(T_STOP_START, T_STOP_END, color="gray", alpha=0.2,
               label="Vehicle stopped (in clutter notch)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("GMTI detection")
    ax.set_title("GMTI detections (1=detected, 0=missed)")
    ax.set_ylim(-0.1, 1.2)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
