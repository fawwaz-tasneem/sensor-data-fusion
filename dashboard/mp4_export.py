"""
MP4 export of a SimulationResult.

The Plotly playback renders inline in the browser; for download, we use
matplotlib's `FuncAnimation` with the `ffmpeg` writer to produce a real
MP4 file. Same animated content: 3D scene with truth/estimate
trajectories, sensors, road map, and tunnel.

`ffmpeg` must be available on PATH. If it isn't, `export_mp4` raises
RuntimeError with a clear message; the dashboard's export callback
catches that and shows a user-facing error.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from dashboard.simulation import SimulationResult


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg to use MP4 export."
        )


def export_mp4(
    result: SimulationResult,
    output_path: str | Path,
    stride: int = 5,
    fps: int = 20,
) -> Path:
    """
    Render the simulation as an MP4 to `output_path`.

    Parameters
    ----------
    result : SimulationResult
    output_path : path the MP4 will be written to.
    stride : keep every N-th time step (default 5) — same downsampling
             as the Plotly playback for consistent timing.
    fps : frames per second in the output video. Real-time playback is
          (1 / dt) * (1 / stride) fps; default 20 makes 1s of video =
          ~5s of simulation at dt=1s, stride=5.

    Returns the output path.
    """
    _check_ffmpeg()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    idx = np.arange(0, len(result.times), stride)
    times = result.times[idx]
    truth = result.truth_positions[idx]
    est = result.estimate_positions[idx]
    sensors = result.sensor_positions[idx]
    detected = result.sensor_detected[idx]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Static trajectories.
    ax.plot(result.truth_positions[:, 0],
            result.truth_positions[:, 1],
            result.truth_positions[:, 2],
            color="green", linewidth=1.2, alpha=0.7, label="Truth")
    ax.plot(result.estimate_positions[:, 0],
            result.estimate_positions[:, 1],
            result.estimate_positions[:, 2],
            color="blue", linewidth=1.0, alpha=0.7, label="Estimate")

    if result.road_nodes is not None:
        ax.plot(result.road_nodes[:, 0],
                result.road_nodes[:, 1],
                result.road_nodes[:, 2],
                "k.-", markersize=2, linewidth=0.5, label="Road")

    if result.tunnel_segments is not None:
        for (p, q) in result.tunnel_segments:
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                    color="dimgray", alpha=0.5, linewidth=0.5)

    # Initial markers (will be updated per frame).
    truth_dot = ax.scatter([truth[0, 0]], [truth[0, 1]], [truth[0, 2]],
                           c="green", s=50, marker="o", label="Truth (now)")
    est_dot = ax.scatter([est[0, 0]], [est[0, 1]], [est[0, 2]],
                         c="blue", s=50, marker="D", label="Estimate (now)")

    sensor_dots = []
    for k in range(sensors.shape[1]):
        color = "purple" if detected[0, k] else "red"
        dot = ax.scatter(
            [sensors[0, k, 0]], [sensors[0, k, 1]], [sensors[0, k, 2]],
            c=color, s=80, marker="^",
        )
        sensor_dots.append(dot)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper right", fontsize=8)
    title = ax.set_title(f"t = {times[0]:.1f} s")

    def update(frame_idx):
        # Update marker positions.
        truth_dot._offsets3d = ([truth[frame_idx, 0]],
                                [truth[frame_idx, 1]],
                                [truth[frame_idx, 2]])
        est_dot._offsets3d = ([est[frame_idx, 0]],
                              [est[frame_idx, 1]],
                              [est[frame_idx, 2]])
        for k, dot in enumerate(sensor_dots):
            dot._offsets3d = ([sensors[frame_idx, k, 0]],
                              [sensors[frame_idx, k, 1]],
                              [sensors[frame_idx, k, 2]])
            dot.set_color("purple" if detected[frame_idx, k] else "red")
        title.set_text(f"t = {times[frame_idx]:.1f} s")
        return [truth_dot, est_dot, *sensor_dots, title]

    anim = animation.FuncAnimation(
        fig, update, frames=len(idx), interval=1000 / fps, blit=False,
    )
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(str(output_path), writer=writer, dpi=100)
    plt.close(fig)
    return output_path


def export_mp4_to_tempfile(result: SimulationResult, **kwargs) -> Path:
    """Convenience: render to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    return export_mp4(result, tmp.name, **kwargs)
