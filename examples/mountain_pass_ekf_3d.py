"""
EKF tracking a vehicle on a 3D mountain pass road, observed by a single
3D radar (range, azimuth, elevation).

This is the marquee example for the project: realistic 3D motion that
genuinely violates the CV assumption (the truth has sinusoidal lateral
and vertical components), tracked through a nonlinear sensor. It's the
ideal stress test before adding IMM.

Outputs a side-by-side plot (truth vs estimate vs measurements) saved
to results/mountain_pass_ekf.png.

Run from project root:
    python examples/mountain_pass_ekf_3d.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; safe for headless runs
import matplotlib.pyplot as plt
import numpy as np

from sdf.core.state import StateDistribution
from sdf.filters import ExtendedKalmanFilter
from sdf.motion_models import ConstantVelocity
from sdf.scenarios import MountainPassTrajectory
from sdf.sensors import RadarSensor
from sdf.simulation import SimulationEngine


def main() -> None:
    # Ground truth: parametrized mountain pass.
    trajectory = MountainPassTrajectory(
        v_kmh=20.0,
        length=10_000.0,
        y_amp=1_000.0,
        z_amp=1_000.0,
    )
    layout = trajectory.layout  # 3D CV-shaped: [x, vx, y, vy, z, vz]

    # 3D constant velocity motion model. The target's true velocity is
    # NOT constant — that mismatch is exactly what stresses the filter
    # and motivates IMM later. The CV process noise must be large enough
    # to absorb the unmodeled lateral and vertical accelerations.
    cv = ConstantVelocity(dim=3, process_noise_std=2.0)

    # 3D radar somewhere off to the side and slightly elevated, so it
    # sees the whole trajectory at non-degenerate geometry.
    radar = RadarSensor(
        sensor_id="radar_3d",
        position=np.array([-2_000.0, -2_000.0, 100.0]),
        range_std=20.0,
        bearing_std=1e-3,
        elevation_std=1e-3,
        detection_prob=1.0,
    )

    # Filter prior: start exactly on truth at t=0 with reasonable uncertainty.
    # (Starting on truth isolates "model mismatch" as the only source of
    #  steady-state error; we'll see whether CV+radar can track the
    #  sinusoidal motion.)
    x0 = trajectory.state_at(0.0)
    initial_estimate = StateDistribution(
        mean=x0.copy(),
        covariance=np.diag([50.0, 5.0, 50.0, 5.0, 50.0, 5.0]) ** 2,
        timestamp=0.0,
        layout=layout,
    )
    ekf = ExtendedKalmanFilter(motion_model=cv, initial_state=initial_estimate)

    engine = SimulationEngine(
        trajectory=trajectory,
        sensors=[radar],
        filter=ekf,
        dt=1.0,        # 1 Hz radar update — typical for surveillance radar
        duration=900.0,  # 15 minutes
        seed=7,
    )
    result = engine.run()

    truth_pos = result.truths[:, layout.position_idx]
    est_pos = result.track.positions()
    pos_err = np.linalg.norm(est_pos - truth_pos, axis=1)

    print(
        f"Mountain Pass + EKF + 3D Radar: {len(result.times)} steps over "
        f"{result.times[-1]:.0f} s"
    )
    print(f"Mean position error: {pos_err.mean():.1f} m")
    print(f"Max  position error: {pos_err.max():.1f} m")
    print(
        "Note: a CV filter cannot perfectly track a sinusoidal trajectory; "
        "this error gives a baseline that IMM will improve on."
    )

    # Plot top-down (xy) view + altitude over time.
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mountain_pass_ekf.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(truth_pos[:, 0], truth_pos[:, 1], label="Truth", linewidth=1.5)
    ax.plot(est_pos[:, 0], est_pos[:, 1], label="EKF estimate", linewidth=1.0, alpha=0.8)
    ax.scatter(radar.position[0], radar.position[1], marker="^", s=80,
               c="red", label="Radar", zorder=5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Top-down view")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(result.times, truth_pos[:, 2], label="Truth z", linewidth=1.5)
    ax.plot(result.times, est_pos[:, 2], label="EKF z", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("z [m]")
    ax.set_title("Altitude profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
