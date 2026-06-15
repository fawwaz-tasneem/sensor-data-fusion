"""
Koch §7.2 + §9.1 example: GMTI radar + road map, with a stopping target.

This reproduces the "stopping target" scenario discussed in Lecture 6
(pages 6, 11, 27) which Koch uses to motivate three things at once:
  1. GMTI gives range-rate, which directly informs velocity.
  2. Stopping targets disappear into the clutter notch (Doppler blindness)
     because |dot{r}| < mdv.
  3. Road information lets the tracker keep going even when sensor
     detections vanish.

We compare three configurations:
  A. Plain EKF + GMTI (no road info, no Doppler-blindness modeling).
  B. Plain EKF + GMTI WITH Doppler blindness simulated (just realistic).
  C. Road-aided EKF + GMTI WITH Doppler blindness — the full Koch setup.

Vehicle moves along a 2D road, then stops for a stretch (entering the
clutter notch where GMTI loses detections), then resumes.
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
from sdf.scenarios import PolygonalRoadMap
from sdf.sensors import DopplerBlindnessOcclusion, GMTIRadarSensor


# ---- Scenario parameters -----------------------------------------------

V_MS = 15.0  # cruising speed, m/s (~ 54 km/h)
T_TOTAL = 200.0  # total simulation time
T_STOP_START = 60.0  # vehicle stops at this time
T_STOP_END = 120.0  # and resumes after this
DT = 1.0  # 1 Hz GMTI scan rate
MDV = 2.0  # Minimum detectable velocity, m/s


# ---- Trajectory: straight road with a stop -----------------------------

def true_state(t: float) -> np.ndarray:
    """[x, vx, y, vy] for a vehicle that moves, stops, then resumes."""
    if t <= T_STOP_START:
        x = V_MS * t
        vx = V_MS
    elif t <= T_STOP_END:
        x = V_MS * T_STOP_START
        vx = 0.0
    else:
        x = V_MS * T_STOP_START + V_MS * (t - T_STOP_END)
        vx = V_MS
    return np.array([x, vx, 0.0, 0.0])


# ---- Build a straight 2D road map -------------------------------------

def build_road_map() -> PolygonalRoadMap:
    """A straight road along the x-axis from x=-100 to x=4000."""
    nodes = np.array([[-100.0, 0.0], [1000.0, 0.0], [2500.0, 0.0], [4000.0, 0.0]])
    return PolygonalRoadMap(nodes, sigma_nodes=2.0)


# ---- Main --------------------------------------------------------------

def main() -> None:
    cv = ConstantVelocity(dim=2, process_noise_std=1.5)
    layout = cv.layout
    road = build_road_map()

    # GMTI sensor without Doppler blindness (configuration A).
    gmti_no_blindness = GMTIRadarSensor(
        sensor_id="gmti_clean",
        position=np.array([2000.0, 3000.0]),  # off to the side
        range_std=15.0,
        bearing_std=2e-3,
        range_rate_std=0.5,
        detection_prob=1.0,
    )

    # GMTI sensor WITH Doppler blindness (configurations B and C).
    blindness = DopplerBlindnessOcclusion(
        sensor_position=np.array([2000.0, 3000.0]),
        mdv=MDV,
    )
    gmti_blind = GMTIRadarSensor(
        sensor_id="gmti_blind",
        position=np.array([2000.0, 3000.0]),
        range_std=15.0,
        bearing_std=2e-3,
        range_rate_std=0.5,
        detection_prob=1.0,
        occlusion_model=blindness,
    )

    # Common initial estimate.
    truth_at_zero = true_state(0.0)
    init = StateDistribution(
        mean=truth_at_zero.copy(),
        covariance=np.diag([100.0, 16.0, 100.0, 16.0]),
        timestamp=0.0,
        layout=layout,
    )

    ekf_a = ExtendedKalmanFilter(motion_model=cv, initial_state=init.copy())
    ekf_b = ExtendedKalmanFilter(motion_model=cv, initial_state=init.copy())
    ekf_c = RoadAidedExtendedKalmanFilter(
        motion_model=cv, initial_state=init.copy(), road_map=road
    )

    # Single rng for both blindness sampling and noise so all 3 see the
    # same target/sensor realizations as far as possible.
    rng = np.random.default_rng(7)

    times = np.arange(0.0, T_TOTAL + DT, DT)
    truths = []
    track_a, track_b, track_c = [], [], []
    detections_b, detections_c = [], []

    for t in times:
        x_true = true_state(t)
        truths.append(x_true)
        if t == 0.0:
            track_a.append(ekf_a.state.mean.copy())
            track_b.append(ekf_b.state.mean.copy())
            track_c.append(ekf_c.state.mean.copy())
            detections_b.append(False)
            detections_c.append(False)
            continue

        # Configuration A: clean GMTI, no blindness.
        m_a = gmti_no_blindness.measure(x_true, layout, t, rng)
        ekf_a.predict(t)
        if m_a is not None:
            ekf_a.update(m_a, gmti_no_blindness)
        track_a.append(ekf_a.state.mean.copy())

        # Configurations B and C: GMTI WITH Doppler blindness.
        # Use a separate rng draw for occlusion sampling so A and B/C
        # see different detections (which is the whole point).
        m_blind = gmti_blind.measure(x_true, layout, t, rng)

        ekf_b.predict(t)
        if m_blind is not None:
            ekf_b.update(m_blind, gmti_blind)
        track_b.append(ekf_b.state.mean.copy())
        detections_b.append(m_blind is not None)

        ekf_c.predict(t)
        if m_blind is not None:
            ekf_c.update(m_blind, gmti_blind)
        else:
            # No reading (target in the clutter notch): fall back on the road.
            ekf_c.update_with_road()
        track_c.append(ekf_c.state.mean.copy())
        detections_c.append(m_blind is not None)

    truths_a = np.array(truths)
    A = np.array(track_a)
    B = np.array(track_b)
    C = np.array(track_c)

    err_a = np.linalg.norm(A[:, [0, 2]] - truths_a[:, [0, 2]], axis=1)
    err_b = np.linalg.norm(B[:, [0, 2]] - truths_a[:, [0, 2]], axis=1)
    err_c = np.linalg.norm(C[:, [0, 2]] - truths_a[:, [0, 2]], axis=1)

    print(f"A) EKF + GMTI clean             : mean err {err_a[1:].mean():.2f} m")
    print(f"B) EKF + GMTI with blindness    : mean err {err_b[1:].mean():.2f} m")
    print(f"C) Road-aided EKF + GMTI w/ b.  : mean err {err_c[1:].mean():.2f} m")
    n_blind_b = sum(1 for d in detections_b[1:] if not d)
    n_blind_c = sum(1 for d in detections_c[1:] if not d)
    print(f"Missed detections in B: {n_blind_b}/{len(detections_b)-1}")
    print(f"Missed detections in C: {n_blind_c}/{len(detections_c)-1}")

    # Plot.
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gmti_with_road_constraint.png")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.plot(times, truths_a[:, 0], "g-", linewidth=1.5, label="Truth x")
    ax.plot(times, A[:, 0], color="orange", alpha=0.8, linewidth=1.0,
            label="A) Clean GMTI")
    ax.plot(times, B[:, 0], color="red", alpha=0.8, linewidth=1.0,
            label="B) GMTI with blindness")
    ax.plot(times, C[:, 0], color="blue", alpha=0.8, linewidth=1.0,
            label="C) Road-aided + GMTI w/ blindness")
    ax.axvspan(T_STOP_START, T_STOP_END, color="gray", alpha=0.2,
               label="Vehicle stopped (in clutter notch)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("x position [m]")
    ax.set_title("Tracking through a stop in the GMTI clutter notch")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(times, err_a, color="orange", alpha=0.8, linewidth=1.0, label="A")
    ax.plot(times, err_b, color="red", alpha=0.8, linewidth=1.0, label="B")
    ax.plot(times, err_c, color="blue", alpha=0.8, linewidth=1.0, label="C")
    ax.axvspan(T_STOP_START, T_STOP_END, color="gray", alpha=0.2)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Position error [m]")
    ax.set_title("Per-step error (lower is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
