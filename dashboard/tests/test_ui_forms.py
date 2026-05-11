"""
Smoke tests for the UI form generator.

We're not testing that Dash renders correctly (that's Dash's concern);
just that the generator functions run without error and produce Dash
components with the expected structure.
"""
from dash import html

from dashboard.components import (
    FILTER_CHOICE,
    MOTION_MODEL_CHOICE,
    OCCLUSION_CHOICE,
    ROAD_MAP,
    SENSOR_LIST,
    TRAJECTORY_CHOICE,
)
from dashboard.ui import (
    form_for_choice,
    form_for_sensor_list,
    form_for_spec,
    scenario_builder_layout,
)


class TestFormForSpec:
    def test_renders_for_each_motion_model(self):
        for key in MOTION_MODEL_CHOICE.keys():
            spec = MOTION_MODEL_CHOICE.get(key)
            form = form_for_spec(spec, section=f"motion_model_{key}")
            assert isinstance(form, html.Div)
            assert form.id["section"] == f"motion_model_{key}"

    def test_renders_road_map(self):
        form = form_for_spec(ROAD_MAP, section="road_map")
        assert isinstance(form, html.Div)


class TestFormForChoice:
    def test_renders_trajectory_choice(self):
        out = form_for_choice(TRAJECTORY_CHOICE, section="trajectory")
        assert isinstance(out, html.Div)

    def test_renders_occlusion_choice_with_none_default(self):
        out = form_for_choice(OCCLUSION_CHOICE, section="occlusion")
        assert isinstance(out, html.Div)
        # Default is "none"; the rendered form should still be a Div.

    def test_renders_filter_choice(self):
        out = form_for_choice(FILTER_CHOICE, section="filter")
        assert isinstance(out, html.Div)


class TestFormForSensorList:
    def test_renders_with_defaults(self):
        out = form_for_sensor_list(SENSOR_LIST, section="sensor_list")
        assert isinstance(out, html.Div)

    def test_renders_with_empty_list(self):
        out = form_for_sensor_list(SENSOR_LIST, section="sensor_list", entries=[])
        assert isinstance(out, html.Div)


class TestScenarioBuilderLayout:
    def test_renders(self):
        out = scenario_builder_layout()
        assert isinstance(out, html.Div)
