"""
Playback view.

Renders a `SimulationResult` as a 3D scene with time-frame animation
plus three side panels (position error, altitude, detection timeline).
The user can scrub, play, or pause via Plotly's built-in animation
controls.

For performance, the 3D scene downsamples frames — at dt=1s and 1800
steps that's 1800 frames, which is several MB of JSON when shipped to
the browser. We keep every N-th frame (default 5) for playback. The
side panels show the full time series so resolution isn't lost where
detail matters.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html

from dashboard.simulation import SimulationResult


def _downsample(result: SimulationResult, stride: int) -> dict:
    """Pick every N-th frame for the animated 3D view."""
    idx = np.arange(0, len(result.times), stride)
    return {
        "times": result.times[idx],
        "truth": result.truth_positions[idx],
        "estimate": result.estimate_positions[idx],
        "sensor_pos": result.sensor_positions[idx],
        "sensor_det": result.sensor_detected[idx],
        "indices": idx,
    }


def build_3d_scene(result: SimulationResult,
                   stride: int = 5) -> go.Figure:
    """
    A 3D Plotly scene of the scenario, animated over time.

    Static traces (added once, visible across all frames):
      - Truth trajectory (line)
      - Estimate trajectory (line)
      - Road map (line of nodes)
      - Tunnel wireframe (segments)

    Animated traces (one per frame):
      - Truth current position (marker)
      - Estimate current position (marker)
      - Sensor positions (markers, recolored by detection state)
    """
    ds = _downsample(result, stride)

    fig = go.Figure()

    # Static: truth and estimate trajectories.
    fig.add_trace(go.Scatter3d(
        x=result.truth_positions[:, 0],
        y=result.truth_positions[:, 1],
        z=result.truth_positions[:, 2],
        mode="lines",
        line=dict(color="green", width=3),
        name="Truth",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=result.estimate_positions[:, 0],
        y=result.estimate_positions[:, 1],
        z=result.estimate_positions[:, 2],
        mode="lines",
        line=dict(color="blue", width=2),
        name="Estimate",
        hoverinfo="skip",
    ))

    # Static: road map.
    if result.road_nodes is not None:
        fig.add_trace(go.Scatter3d(
            x=result.road_nodes[:, 0],
            y=result.road_nodes[:, 1],
            z=result.road_nodes[:, 2],
            mode="lines+markers",
            line=dict(color="black", width=1),
            marker=dict(size=2, color="black"),
            name="Road map",
            hoverinfo="skip",
        ))

    # Static: tunnel wireframe (segments encoded as one long Scatter3d
    # with None separators, the Plotly idiom for unconnected line segments).
    if result.tunnel_segments is not None:
        xs, ys, zs = [], [], []
        for (p, q) in result.tunnel_segments:
            xs.extend([p[0], q[0], None])
            ys.extend([p[1], q[1], None])
            zs.extend([p[2], q[2], None])
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color="dimgray", width=2),
            name="Tunnel",
            hoverinfo="skip",
            opacity=0.6,
        ))

    # Animated: truth/estimate current marker + sensor markers.
    # We add placeholder traces for the FIRST frame; the frames list
    # below provides updates.
    first = 0
    fig.add_trace(go.Scatter3d(
        x=[ds["truth"][first, 0]],
        y=[ds["truth"][first, 1]],
        z=[ds["truth"][first, 2]],
        mode="markers",
        marker=dict(size=6, color="green", symbol="circle"),
        name="Truth (now)",
    ))
    fig.add_trace(go.Scatter3d(
        x=[ds["estimate"][first, 0]],
        y=[ds["estimate"][first, 1]],
        z=[ds["estimate"][first, 2]],
        mode="markers",
        marker=dict(size=6, color="blue", symbol="diamond"),
        name="Estimate (now)",
    ))
    # Sensors: one trace per sensor, recolored by detection state.
    for k, sid in enumerate(result.sensor_ids):
        # Color sensors red when they're NOT detecting (blocked).
        detected = ds["sensor_det"][first, k]
        color = "purple" if detected else "red"
        fig.add_trace(go.Scatter3d(
            x=[ds["sensor_pos"][first, k, 0]],
            y=[ds["sensor_pos"][first, k, 1]],
            z=[ds["sensor_pos"][first, k, 2]],
            mode="markers+text",
            marker=dict(size=8, color=color, symbol="diamond-open"),
            text=[sid],
            textposition="top center",
            name=sid,
            hoverinfo="text",
        ))

    # Indices of the animated traces (last 2 + n_sensors).
    n_sensors = len(result.sensor_ids)
    n_static_traces = sum([
        1,  # truth line
        1,  # estimate line
        1 if result.road_nodes is not None else 0,
        1 if result.tunnel_segments is not None else 0,
    ])
    animated_start = n_static_traces
    truth_marker_idx = animated_start
    estimate_marker_idx = animated_start + 1
    sensor_marker_indices = list(range(animated_start + 2,
                                       animated_start + 2 + n_sensors))

    # Build frames.
    frames = []
    for fi, idx in enumerate(ds["indices"]):
        frame_traces = []
        # truth marker
        frame_traces.append(go.Scatter3d(
            x=[ds["truth"][fi, 0]],
            y=[ds["truth"][fi, 1]],
            z=[ds["truth"][fi, 2]],
        ))
        # estimate marker
        frame_traces.append(go.Scatter3d(
            x=[ds["estimate"][fi, 0]],
            y=[ds["estimate"][fi, 1]],
            z=[ds["estimate"][fi, 2]],
        ))
        # sensor markers
        for k in range(n_sensors):
            detected = bool(ds["sensor_det"][fi, k])
            color = "purple" if detected else "red"
            frame_traces.append(go.Scatter3d(
                x=[ds["sensor_pos"][fi, k, 0]],
                y=[ds["sensor_pos"][fi, k, 1]],
                z=[ds["sensor_pos"][fi, k, 2]],
                marker=dict(size=8, color=color, symbol="diamond-open"),
            ))
        frames.append(go.Frame(
            data=frame_traces,
            traces=[truth_marker_idx, estimate_marker_idx, *sensor_marker_indices],
            name=str(fi),
        ))

    fig.frames = frames

    fig.update_layout(
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=520,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.05, "y": 0.05, "xanchor": "left", "yanchor": "bottom",
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": 50, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    }],
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                },
            ],
        }],
        sliders=[{
            "active": 0,
            "x": 0.15, "y": 0.05, "xanchor": "left", "yanchor": "bottom",
            "len": 0.8,
            "currentvalue": {"prefix": "t = ", "suffix": " s",
                             "visible": True},
            "steps": [
                {
                    "args": [[str(fi)], {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                    }],
                    "label": f"{ds['times'][fi]:.0f}",
                    "method": "animate",
                }
                for fi in range(len(ds["indices"]))
            ],
        }],
    )

    return fig


def build_error_panel(result: SimulationResult) -> go.Figure:
    err = np.linalg.norm(
        result.truth_positions - result.estimate_positions, axis=1
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.times, y=err,
        mode="lines", line=dict(color="red", width=1.2),
        name="Position error",
        hovertemplate="t=%{x:.0f}s<br>err=%{y:.1f}m<extra></extra>",
    ))
    fig.update_layout(
        title="Position error [m]",
        xaxis_title="t [s]", yaxis_title="error [m]",
        margin=dict(l=40, r=10, t=30, b=30),
        height=200, showlegend=False,
    )
    return fig


def build_altitude_panel(result: SimulationResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.times, y=result.truth_positions[:, 2],
        mode="lines", line=dict(color="green", width=1.2), name="Truth z",
    ))
    fig.add_trace(go.Scatter(
        x=result.times, y=result.estimate_positions[:, 2],
        mode="lines", line=dict(color="blue", width=1.2), name="Estimate z",
    ))
    fig.update_layout(
        title="Altitude [m]",
        xaxis_title="t [s]", yaxis_title="z [m]",
        margin=dict(l=40, r=10, t=30, b=30),
        height=200, showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2),
    )
    return fig


def build_detection_panel(result: SimulationResult) -> go.Figure:
    fig = go.Figure()
    n_sensors = len(result.sensor_ids)
    for k, sid in enumerate(result.sensor_ids):
        det = result.sensor_detected[:, k].astype(float)
        # Plot detected (1) vs missed (0) as a step series, offset by sensor.
        offset = k * 1.2
        fig.add_trace(go.Scatter(
            x=result.times, y=det + offset,
            mode="lines", line=dict(color="purple", width=1.5, shape="hv"),
            name=sid,
        ))
    fig.update_layout(
        title="Sensor detection (1 = received, 0 = missed)",
        xaxis_title="t [s]",
        yaxis=dict(
            tickmode="array",
            tickvals=[k * 1.2 + 0.5 for k in range(n_sensors)],
            ticktext=result.sensor_ids,
        ),
        margin=dict(l=80, r=10, t=30, b=30),
        height=200,
        showlegend=False,
    )
    return fig


def playback_layout(result: Optional[SimulationResult]) -> html.Div:
    """Top-level playback layout: 3D scene + 3 side panels."""
    if result is None:
        return html.Div(
            "Configure a scenario on the left and click 'Run simulation'.",
            style={"padding": "40px", "textAlign": "center",
                   "color": "#888", "fontSize": "1.1em"},
        )

    return html.Div(
        [
            html.Div(
                dcc.Graph(figure=build_3d_scene(result),
                          id="playback-3d"),
                style={"flex": "2", "minWidth": "0"},
            ),
            html.Div(
                [
                    dcc.Graph(figure=build_error_panel(result)),
                    dcc.Graph(figure=build_altitude_panel(result)),
                    dcc.Graph(figure=build_detection_panel(result)),
                ],
                style={"flex": "1", "minWidth": "0",
                       "display": "flex", "flexDirection": "column"},
            ),
        ],
        style={"display": "flex", "flexDirection": "row",
               "gap": "8px", "width": "100%"},
    )
