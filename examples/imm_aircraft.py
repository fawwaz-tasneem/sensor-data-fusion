"""
IMM tracking of a maneuvering aircraft with two stationary radars.

The target flies a five-segment trajectory:
  0-30 s     straight along +x
  30-60 s    turn left (omega = +0.05 rad/s)
  60-90 s    straight
  90-120 s   turn right (omega = -0.05 rad/s)
  120-180 s  straight

We compare three configurations:
  A. Plain EKF + CV motion model
  B. Plain EKF + CT model with omega=0 (basically CV)
  C. IMM = CV + CT-left + CT-right

Two stationary radars observe the target. This example uses the 2D (x, y)
projection of the radars at (0, 10000) and (10000, 0); the small z offset
(100 m) is dropped for a clean 2D demonstration of IMM mode switching.

Outputs:
  results/imm_aircraft.png - top-down trajectory + per-step error +
                              mode probabilities over time
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sdf.core.state import StateDistribution, StateLayout
from sdf.filters import ExtendedKalmanFilter, IMMFilter, KalmanFilter
from sdf.motion_models import ConstantVelocity, CoordinatedTurn
from sdf.sensors import RadarSensor


def true_state(t: float) -> np.ndarray:
    """Aircraft trajectory: straight, left turn, straight, right turn, straight."""
    speed = 100.0  # m/s

    if t <= 30.0:
        # Straight along +x.
        x = 1000.0 + speed * t
        y = 1000.0
        vx, vy = speed, 0.0
        return np.array([x, vx, y, vy])

    # Position at end of segment 1.
    x1 = 1000.0 + speed * 30.0
    y1 = 1000.0
    vx1, vy1 = speed, 0.0

    if t <= 60.0:
        # Left turn (omega = +0.05).
        omega = 0.05
        tau = t - 30.0
        c = np.cos(omega * tau)
        s = np.sin(omega * tau)
        # Velocity rotation: [vx', vy'] = R(omega tau) [vx, vy]
        vx = c * vx1 - s * vy1
        vy = s * vx1 + c * vy1
        # Position by integrating: x = x1 + (vx1 sin + vy1 (cos-1))/(-omega... )
        # Equivalent: x = x1 + vx1 sin(om tau)/om - vy1 (1-cos(om tau))/om
        x = x1 + vx1 * s / omega - vy1 * (1 - c) / omega
        y = y1 + vx1 * (1 - c) / omega + vy1 * s / omega
        return np.array([x, vx, y, vy])

    # Position at end of segment 2.
    omega = 0.05
    s30 = np.sin(omega * 30.0)
    c30 = np.cos(omega * 30.0)
    vx2 = c30 * vx1 - s30 * vy1
    vy2 = s30 * vx1 + c30 * vy1
    x2 = x1 + vx1 * s30 / omega - vy1 * (1 - c30) / omega
    y2 = y1 + vx1 * (1 - c30) / omega + vy1 * s30 / omega

    if t <= 90.0:
        # Straight along the new heading.
        tau = t - 60.0
        x = x2 + vx2 * tau
        y = y2 + vy2 * tau
        return np.array([x, vx2, y, vy2])

    # End of segment 3.
    x3 = x2 + vx2 * 30.0
    y3 = y2 + vy2 * 30.0

    if t <= 120.0:
        # Right turn (omega = -0.05).
        omega = -0.05
        tau = t - 90.0
        c = np.cos(omega * tau)
        s = np.sin(omega * tau)
        vx = c * vx2 - s * vy2
        vy = s * vx2 + c * vy2
        x = x3 + vx2 * s / omega - vy2 * (1 - c) / omega
        y = y3 + vx2 * (1 - c) / omega + vy2 * s / omega
        return np.array([x, vx, y, vy])

    # End of segment 4.
    omega = -0.05
    s30 = np.sin(omega * 30.0)
    c30 = np.cos(omega * 30.0)
    vx4 = c30 * vx2 - s30 * vy2
    vy4 = s30 * vx2 + c30 * vy2
    x4 = x3 + vx2 * s30 / omega - vy2 * (1 - c30) / omega
    y4 = y3 + vx2 * (1 - c30) / omega + vy2 * s30 / omega

    # Segment 5: straight along new heading.
    tau = t - 120.0
    x = x4 + vx4 * tau
    y = y4 + vy4 * tau
    return np.array([x, vx4, y, vy4])


def main() -> None:
    # State layout: [x, vx, y, vy] (dim 4, 2D).
    layout = StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))

    # Two stationary radars at (0, 10000) and (10000, 0) - the 2D projection
    # of the canonical 3D sensor positions.
    radar_a = RadarSensor(
        sensor_id="radar_a",
        position=np.array([0.0, 10_000.0]),
        range_std=30.0,
        bearing_std=3e-3,
        detection_prob=1.0,
    )
    radar_b = RadarSensor(
        sensor_id="radar_b",
        position=np.array([10_000.0, 0.0]),
        range_std=30.0,
        bearing_std=3e-3,
        detection_prob=1.0,
    )
    sensors = [radar_a, radar_b]

    # Common initial estimate (slightly off truth).
    x_init_true = true_state(0.0)
    init = StateDistribution(
        mean=x_init_true + np.array([10.0, 0.5, 10.0, 0.5]),
        covariance=np.diag([100.0, 25.0, 100.0, 25.0]),
        timestamp=0.0,
        layout=layout,
    )

    # Configuration A: pure CV filter.
    cv = ConstantVelocity(dim=2, process_noise_std=2.0)
    plain_cv = KalmanFilter(motion_model=cv, initial_state=init.copy())

    # Configuration B: IMM with CV + CT-left + CT-right.
    cv_for_imm = ConstantVelocity(dim=2, process_noise_std=2.0)
    ct_left = CoordinatedTurn(omega=+0.05, process_noise_std=2.0)
    ct_right = CoordinatedTurn(omega=-0.05, process_noise_std=2.0)
    f_cv = KalmanFilter(motion_model=cv_for_imm, initial_state=init.copy())
    f_left = ExtendedKalmanFilter(motion_model=ct_left, initial_state=init.copy())
    f_right = ExtendedKalmanFilter(motion_model=ct_right, initial_state=init.copy())

    imm = IMMFilter(
        filters=[f_cv, f_left, f_right],
        transition_matrix=np.array([
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
        ]),
        mode_probs=np.array([1 / 3, 1 / 3, 1 / 3]),
    )

    dt = 0.5
    duration = 180.0
    times = np.arange(0.0, duration + dt, dt)
    rng = np.random.default_rng(7)

    truths, cv_track, imm_track, mode_probs_history = [], [], [], []
    for t in times:
        x_true = true_state(t)
        truths.append(x_true)
        if t == 0.0:
            cv_track.append(plain_cv.state.mean.copy())
            imm_track.append(imm.state.mean.copy())
            mode_probs_history.append(imm.mode_probabilities)
            continue

        plain_cv.predict(t)
        imm.predict(t)

        for sensor in sensors:
            m = sensor.measure(x_true, layout, t, rng)
            if m is not None:
                plain_cv.update(m, sensor)
                imm.update(m, sensor)

        cv_track.append(plain_cv.state.mean.copy())
        imm_track.append(imm.state.mean.copy())
        mode_probs_history.append(imm.mode_probabilities)

    truths_a = np.array(truths)
    cv_a = np.array(cv_track)
    imm_a = np.array(imm_track)
    mode_probs_a = np.array(mode_probs_history)

    truth_pos = truths_a[:, [0, 2]]
    cv_pos = cv_a[:, [0, 2]]
    imm_pos = imm_a[:, [0, 2]]
    cv_err = np.linalg.norm(cv_pos - truth_pos, axis=1)
    imm_err = np.linalg.norm(imm_pos - truth_pos, axis=1)

    print(f"Plain CV mean position error: {cv_err[1:].mean():.2f} m  "
          f"(max {cv_err[1:].max():.2f})")
    print(f"IMM      mean position error: {imm_err[1:].mean():.2f} m  "
          f"(max {imm_err[1:].max():.2f})")

    # Plot.
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "imm_aircraft.png")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])

    # Top-down trajectory.
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(truth_pos[:, 0], truth_pos[:, 1], color="green", linewidth=1.5,
            label="Truth", alpha=0.85)
    ax.plot(cv_pos[:, 0], cv_pos[:, 1], color="red", linewidth=1.0,
            label="Plain CV", alpha=0.8)
    ax.plot(imm_pos[:, 0], imm_pos[:, 1], color="blue", linewidth=1.0,
            label="IMM (CV + CT-left + CT-right)", alpha=0.8)
    for sensor, name in [(radar_a, "Radar A"), (radar_b, "Radar B")]:
        ax.scatter(sensor.position[0], sensor.position[1], marker="^",
                   s=80, c="purple", zorder=5)
        ax.annotate(name, (sensor.position[0], sensor.position[1]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Maneuvering aircraft: trajectory")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    # Per-step error.
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(times, cv_err, color="red", linewidth=1.0, label="Plain CV", alpha=0.85)
    ax.plot(times, imm_err, color="blue", linewidth=1.0, label="IMM", alpha=0.85)
    for t_seg, label in [(30, "Left turn"), (60, "Straight"),
                          (90, "Right turn"), (120, "Straight")]:
        ax.axvline(t_seg, color="gray", linestyle="--", linewidth=0.5)
        ax.text(t_seg, ax.get_ylim()[1] * 0.95, label, rotation=90,
                fontsize=7, ha="right", va="top", color="gray")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Position error [m]")
    ax.set_title("Per-step position error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mode probabilities over time.
    ax = fig.add_subplot(gs[1, :])
    ax.fill_between(times, 0, mode_probs_a[:, 0], color="gray", alpha=0.6,
                    label="CV (omega=0)")
    ax.fill_between(times, mode_probs_a[:, 0],
                    mode_probs_a[:, 0] + mode_probs_a[:, 1],
                    color="blue", alpha=0.6, label="CT-left (omega=+0.05)")
    ax.fill_between(times,
                    mode_probs_a[:, 0] + mode_probs_a[:, 1],
                    mode_probs_a[:, 0] + mode_probs_a[:, 1] + mode_probs_a[:, 2],
                    color="red", alpha=0.6, label="CT-right (omega=-0.05)")
    # Annotate true mode segments.
    ax.axvspan(0, 30, alpha=0.0)
    for x_start, x_end, lbl, color in [
        (0, 30, "Straight", "darkgreen"),
        (30, 60, "Left turn", "darkblue"),
        (60, 90, "Straight", "darkgreen"),
        (90, 120, "Right turn", "darkred"),
        (120, 180, "Straight", "darkgreen"),
    ]:
        ax.axvline(x_end, color="black", linewidth=0.5, alpha=0.5)
        ax.text((x_start + x_end) / 2, 1.05, lbl, ha="center", fontsize=8,
                color=color)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Mode probability")
    ax.set_title("IMM mode probabilities (truth segments labeled on top)")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(times[0], times[-1])
    ax.legend(loc="center right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
