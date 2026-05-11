"""Dashboard UI: pure functions producing Dash layout from schemas."""
from dashboard.ui.forms import (
    form_for_choice,
    form_for_sensor_list,
    form_for_spec,
    scenario_builder_layout,
)

__all__ = [
    "form_for_spec",
    "form_for_choice",
    "form_for_sensor_list",
    "scenario_builder_layout",
]
