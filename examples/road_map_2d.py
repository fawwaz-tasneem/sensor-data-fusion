"""
Koch §9.1 example: road-aided EKF tracking on a 2D mountain pass road,
using the parameters from Lecture 6.

Lecture parameters:
  v       = 20 km/h
  ax      = 10 km    (x-axis range, characteristic length)
  ay = az = 1 km     (sinusoid amplitudes)
  t       in [0, ax / v]

Trajectory (mountain pass):
  x(t) = v t
  y(t) = ay sin(4 pi v t / ax)
  z(t) = az sin(  pi v t / ax)

For a 2D demo we drop z and project onto the (x, y) plane.

What this demonstrates:
  1. A *sparse* polygonal road map (~30 nodes) approximates the smooth
     sinusoidal road. Discretization error sigma_d_m enters as a
     non-zero variance on each segment.
  2. A radar with high range/bearing noise tracks the vehicle.
  3. The road-aided EKF outperforms a plain EKF on the same data,
     reducing cross-track error specifically.

Outputs:
  results/koch_road_map_2d.png — top-down plot with road, truth, and
                                  both EKF estimates.
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
from sdf.sensors import RadarSensor


# ---- Lecture parameters ------------------------------------------------

V_KMH = 20.0
AX = 10_000.0  # m
AY = 1_000.0  # m
V_MS = V_KMH / 3.6


def true_position_2d(t: float) -> np.ndarray:
    """The lecture's mountain pass trajectory, projected to 2D."""
    x = V_MS * t
    y = AY * np.sin((4 * np.pi * V_MS / AX) * t)
    return np.array([x, y])


def true_velocity_2d(t: float) -> np.ndarray:
    """Analytic derivative of the position function."""
    vx = V_MS
    vy = AY * (4 * np.pi * V_MS / AX) * np.cos((4 * np.pi * V_MS / AX) * t)
    return np.array([vx, vy])


def true_state_2d(t: float) -> np.ndarray:
    """Full CV-shaped state [x, vx, y, vy] at time t."""
    p = true_position_2d(t)
    v = true_velocity_2d(t)
    return np.array([p[0], v[0], p[1], v[1]])


# ---- Build a sparse polygonal map ------------------------------------

def build_road_map(n_nodes: int = 30, t_start: float = 0.0, t_end: float | None = None
                   ) -> PolygonalRoadMap:
    """
    Sample the truth at sparse times, then declare the polygonal map's
    arc lengths to be the *true* arc lengths along the smooth curve.

    The chord between consecutive nodes is shorter than the actual arc
    length the vehicle travels, so discretization error sigma_d > 0
    on every segment — exactly the effect Koch's lecture describes.
    """
    if t_end is None:
        t_end = AX / V_MS  # the lecture's t in [0, ax/v]
    times = np.linspace(t_start, t_end, n_nodes)
    nodes = np.array([true_position_2d(t) for t in times])

    # Compute the *true* arc length along the smooth curve between
    # consecutive sampled times via dense numerical integration.
    arc_lengths = [0.0]
    for i in range(1, n_nodes):
        ts = np.linspace(times[i - 1], times[i], 200)
        pts = np.array([true_position_2d(t) for t in ts])
        seg_arc = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        arc_lengths.append(arc_lengths[-1] + seg_arc)
    arc_lengths_arr = np.array(arc_lengths)

    # Mapping accuracy: assume nodes are surveyed to within 5 m.
    return PolygonalRoadMap(
        nodes=nodes, arc_lengths=arc_lengths_arr, sigma_nodes=5.0
    )


# ---- Main --------------------------------------------------------------

def main() -> None:
    cv = ConstantVelocity(dim=2, process_noise_std=2.0)
    layout = cv.layout

    # Build the sparse road map. With 30 nodes over a 10 km road, we get
    # ~330 m chords, plenty of discretization error around the sine peaks.
    road = build_road_map(n_nodes=30)
    print(f"Road map: {len(road)} segments, total arc length "
          f"{road.total_length():.0f} m")
    max_disc_err = max(seg.sigma_disc for seg in road.segments)
    print(f"Max discretization error sigma_d on any segment: {max_disc_err:.2f} m")

    # Radar far off to the side. We use a noisy radar (large range/bearing
    # std) and a sparse update interval — the lecture's point is that road
    # information is most valuable when the sensor itself is uncertain.
    radar = RadarSensor(
        sensor_id="radar_1",
        position=np.array([5000.0, -5000.0]),
        range_std=80.0,
        bearing_std=8e-3,
        detection_prob=1.0,
    )

    # Both filters start at the same prior, deliberately offset off-road
    # by ~50 m to make the road update do something visible in the plot.
    truth_at_zero = true_state_2d(0.0)
    prior_mean = truth_at_zero.copy()
    prior_mean[2] += 50.0  # y: 50 m off-road
    initial_estimate = StateDistribution(
        mean=prior_mean,
        covariance=np.diag([400.0, 16.0, 400.0, 16.0]),
        timestamp=0.0,
        layout=layout,
    )

    plain_ekf = ExtendedKalmanFilter(motion_model=cv, initial_state=initial_estimate.copy())
    aided_ekf = RoadAidedExtendedKalmanFilter(
        motion_model=cv, initial_state=initial_estimate.copy(), road_map=road
    )

    # Simulate.
    dt = 1.0
    duration = AX / V_MS  # the lecture's full t-range
    times = np.arange(0.0, duration + dt, dt)
    rng = np.random.default_rng(42)

    truths, plain_track, aided_track = [], [], []
    for t in times:
        x_true = true_state_2d(t)
        truths.append(x_true)
        if t == 0.0:
            plain_track.append(plain_ekf.state.mean.copy())
            aided_track.append(aided_ekf.state.mean.copy())
            continue
        m = radar.measure(x_true, layout, t, rng)
        plain_ekf.predict(t)
        plain_ekf.update(m, radar)
        aided_ekf.predict(t)
        aided_ekf.update(m, radar)
        # Apply the road update at every step.
        aided_ekf.update_with_road()
        plain_track.append(plain_ekf.state.mean.copy())
        aided_track.append(aided_ekf.state.mean.copy())

    truths_a = np.array(truths)
    plain_a = np.array(plain_track)
    aided_a = np.array(aided_track)

    # Per-step position error (skip t=0 which is the initial prior).
    plain_err = np.linalg.norm(plain_a[:, [0, 2]] - truths_a[:, [0, 2]], axis=1)
    aided_err = np.linalg.norm(aided_a[:, [0, 2]] - truths_a[:, [0, 2]], axis=1)
    print(
        f"Plain EKF mean position error (post-init): {plain_err[1:].mean():.2f} m  "
        f"(max {plain_err[1:].max():.2f})"
    )
    print(
        f"Aided EKF mean position error (post-init): {aided_err[1:].mean():.2f} m  "
        f"(max {aided_err[1:].max():.2f})"
    )

    # Plot.
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "koch_road_map_2d.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    # Smooth truth curve.
    smooth_t = np.linspace(0, duration, 1000)
    smooth_pos = np.array([true_position_2d(t) for t in smooth_t])
    ax.plot(smooth_pos[:, 0], smooth_pos[:, 1], color="green", label="Truth (smooth)",
            linewidth=1.0, alpha=0.7)
    # Polygonal map.
    map_pts = road.nodes
    ax.plot(map_pts[:, 0], map_pts[:, 1], "ko-", markersize=3, linewidth=0.7,
            label=f"Road map ({len(road)} segments)")
    # Estimates.
    ax.plot(plain_a[:, 0], plain_a[:, 2], color="red", linewidth=1.0,
            label="Plain EKF", alpha=0.8)
    ax.plot(aided_a[:, 0], aided_a[:, 2], color="blue", linewidth=1.0,
            label="Road-aided EKF", alpha=0.8)
    ax.scatter(radar.position[0], radar.position[1], marker="^",
               s=80, c="purple", label="Radar", zorder=5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Koch road-aided tracking — top-down view")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(times, plain_err, color="red", label="Plain EKF", linewidth=1.0)
    ax.plot(times, aided_err, color="blue", label="Road-aided EKF", linewidth=1.0)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Position error [m]")
    ax.set_title("Tracking error over time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
