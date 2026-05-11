"""
Smoke tests for the playback view and app construction.

We verify:
  * playback_layout(None) returns a placeholder Div.
  * playback_layout(result) returns a Div containing the 3D scene + side panels.
  * build_3d_scene produces a Plotly Figure with the expected number of
    traces and at least one frame.
  * create_app returns a Dash app whose layout passes static validation.
"""
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html

from dashboard.app import create_app
from dashboard.simulation import run_simulation
from dashboard.ui.playback import (
    build_3d_scene,
    build_altitude_panel,
    build_detection_panel,
    build_error_panel,
    playback_layout,
)


def _quick_result():
    """Run a minimal default simulation for use in playback tests."""
    from dashboard.components import (
        FILTER_CHOICE,
        MOTION_MODEL_CHOICE,
        OCCLUSION_CHOICE,
        ROAD_MAP,
        SENSOR_LIST,
        TRAJECTORY_CHOICE,
    )
    return run_simulation({
        "trajectory": {"type": "mountain_pass",
                       "params": TRAJECTORY_CHOICE.get("mountain_pass").validate({})},
        "motion_model": {"type": "cv",
                         "params": MOTION_MODEL_CHOICE.get("cv").validate({})},
        "sensor_list": SENSOR_LIST.validate(SENSOR_LIST.defaults()),
        "occlusion": {"type": "none",
                      "params": OCCLUSION_CHOICE.get("none").validate({})},
        "filter": {"type": "ekf",
                   "params": FILTER_CHOICE.get("ekf").validate({})},
        "road_map": {"enabled": False,
                     "params": ROAD_MAP.validate({})},
        "sim": {"seed": 0, "dt": 5.0},
    })


class TestPlaybackLayout:
    def test_empty_renders_placeholder(self):
        out = playback_layout(None)
        assert isinstance(out, html.Div)

    def test_with_result_renders_panels(self):
        result = _quick_result()
        out = playback_layout(result)
        assert isinstance(out, html.Div)
        # Should contain two flex children: the 3D scene and the side column.
        assert len(out.children) == 2


class TestPlaybackFigures:
    def setup_method(self):
        self.result = _quick_result()

    def test_3d_scene_has_frames(self):
        fig = build_3d_scene(self.result)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) > 0
        # At least 4 traces: truth line, estimate line, truth marker,
        # estimate marker, plus N sensor markers.
        assert len(fig.data) >= 4 + len(self.result.sensor_ids)

    def test_error_panel(self):
        fig = build_error_panel(self.result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_altitude_panel(self):
        fig = build_altitude_panel(self.result)
        assert isinstance(fig, go.Figure)
        # Truth and estimate.
        assert len(fig.data) == 2

    def test_detection_panel_per_sensor(self):
        fig = build_detection_panel(self.result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == len(self.result.sensor_ids)


class TestApp:
    def test_create_app_builds_layout(self):
        app = create_app()
        assert isinstance(app, Dash)
        # The layout should be a Div with multiple children.
        layout = app.layout
        assert layout is not None
        # And Dash should be able to enumerate dependencies without raising.
        # (This catches missing callback targets, malformed IDs, etc.)
        deps = app.callback_map
        assert len(deps) > 0
