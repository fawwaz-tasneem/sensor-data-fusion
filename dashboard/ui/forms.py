"""
UI form generator.

Pure functions that turn a `ComponentSpec` / `ComponentChoice` /
`SensorListSpec` into Dash layout. No callbacks — those live in app.py.
The forms emit pattern-matching IDs of the form

    {"section": <section>, "field": <field>, "index": <index>}

so that callbacks can match all fields belonging to a given section, or
the same field across all sections, etc.

For a ComponentSpec named under section "trajectory":

    {"section": "trajectory", "field": "v_kmh", "index": 0}

For a sensor-list entry's field, the index is the entry's position in
the list:

    {"section": "sensor_list", "field": "sensor_id", "index": 0}

Vector parameters render as N inputs, each tagged with `"slot": k`:

    {"section": "trajectory", "field": "position", "index": 0, "slot": 1}
"""
from __future__ import annotations

from typing import Any, Optional

from dash import dcc, html

from dashboard.components import (
    FILTER_CHOICE,
    MOTION_MODEL_CHOICE,
    OCCLUSION_CHOICE,
    ROAD_MAP,
    SENSOR_LIST,
    TRAJECTORY_CHOICE,
    SensorListSpec,
)
from dashboard.schema import (
    ComponentChoice,
    ComponentSpec,
    ParameterSpec,
)


def _field_id(section: str, field: str, index: int = 0,
              slot: Optional[int] = None) -> dict:
    """Pattern-matching ID for a single form field."""
    out = {"section": section, "field": field, "index": index}
    if slot is not None:
        out["slot"] = slot
    return out


def _input_for_scalar(
    parameter: ParameterSpec,
    section: str,
    index: int,
    initial: Any,
) -> Any:
    """Build a single Dash input control for a scalar parameter."""
    if parameter.choices is not None:
        return dcc.Dropdown(
            id=_field_id(section, parameter.name, index),
            options=[
                {"label": label, "value": value}
                for label, value in parameter.choices
            ],
            value=initial,
            clearable=False,
            style={"width": "100%"},
        )
    if parameter.kind is bool:
        return dcc.Checklist(
            id=_field_id(section, parameter.name, index),
            options=[{"label": "", "value": True}],
            value=[True] if initial else [],
        )
    if parameter.kind is str:
        return dcc.Input(
            id=_field_id(section, parameter.name, index),
            type="text",
            value=initial,
            style={"width": "100%"},
        )
    # Numeric kinds: float, int.
    return dcc.Input(
        id=_field_id(section, parameter.name, index),
        type="number",
        value=initial,
        min=parameter.min,
        max=parameter.max,
        step=parameter.step,
        style={"width": "100%"},
    )


def _row_for_parameter(
    parameter: ParameterSpec,
    section: str,
    index: int,
    values: dict[str, Any],
) -> html.Div:
    """One labelled row in a parameter form."""
    label_text = parameter.description or parameter.name
    if parameter.unit:
        label_text = f"{label_text} [{parameter.unit}]"

    if parameter.length > 1:
        initial = values.get(parameter.name, parameter.default)
        controls = html.Div(
            [
                dcc.Input(
                    id=_field_id(section, parameter.name, index, slot=k),
                    type="number",
                    value=initial[k],
                    step=parameter.step,
                    style={"width": "100%"},
                )
                for k in range(parameter.length)
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": f"repeat({parameter.length}, 1fr)",
                "gap": "4px",
            },
        )
    else:
        initial = values.get(parameter.name, parameter.default)
        controls = _input_for_scalar(parameter, section, index, initial)

    return html.Div(
        [
            html.Label(label_text, style={"fontSize": "0.85em",
                                          "color": "#555"}),
            controls,
        ],
        style={"marginBottom": "10px"},
    )


def form_for_spec(
    spec: ComponentSpec,
    section: str,
    index: int = 0,
    values: Optional[dict[str, Any]] = None,
) -> html.Div:
    """
    A parameter form for one ComponentSpec.

    `values` overrides defaults for individual fields; anything missing
    falls back to the spec's defaults.
    """
    if values is None:
        values = spec.defaults()
    rows = [_row_for_parameter(p, section, index, values) for p in spec.parameters]
    if spec.description:
        rows.insert(0, html.Div(
            spec.description,
            style={"fontSize": "0.8em", "color": "#777",
                   "fontStyle": "italic", "marginBottom": "10px"},
        ))
    return html.Div(rows, id={"section": section, "kind": "form",
                              "index": index})


def form_for_choice(
    choice: ComponentChoice,
    section: str,
    selected_key: Optional[str] = None,
    values: Optional[dict[str, Any]] = None,
) -> html.Div:
    """
    Dropdown + parameter form for a ComponentChoice.

    The dropdown emits `{"section": section, "field": "__choice__", "index": 0}`
    which a callback in app.py listens to and re-renders the form below.
    """
    if selected_key is None:
        selected_key = choice.default_key
    selected_spec = choice.get(selected_key)
    form = form_for_spec(selected_spec, section, index=0, values=values)
    return html.Div(
        [
            html.Label(choice.label,
                       style={"fontWeight": "bold", "fontSize": "0.95em"}),
            dcc.Dropdown(
                id=_field_id(section, "__choice__", 0),
                options=[
                    {"label": choice.options[k].label, "value": k}
                    for k in choice.keys()
                ],
                value=selected_key,
                clearable=False,
            ),
            html.Div(form,
                     id={"section": section, "kind": "choice_form_holder",
                         "index": 0},
                     style={"marginTop": "8px",
                            "paddingLeft": "8px",
                            "borderLeft": "2px solid #ddd"}),
        ],
        style={"marginBottom": "16px"},
    )


def form_for_sensor_list(
    spec: SensorListSpec,
    section: str = "sensor_list",
    entries: Optional[list[dict[str, Any]]] = None,
) -> html.Div:
    """
    Variable-length sensor list with add/remove controls.

    Each entry renders as a card with a dropdown (sensor type) and a
    parameter form. The list itself sits inside a container that
    add/remove buttons mutate via callbacks in app.py.
    """
    if entries is None:
        entries = spec.defaults()

    cards = []
    for i, entry in enumerate(entries):
        sensor_spec = spec.choice.get(entry["type"])
        cards.append(html.Div(
            [
                html.Div([
                    html.Span(f"Sensor #{i + 1}",
                              style={"fontWeight": "bold"}),
                    html.Button(
                        "Remove",
                        id={"section": section, "field": "__remove__",
                            "index": i},
                        n_clicks=0,
                        style={"float": "right",
                               "padding": "2px 8px",
                               "fontSize": "0.8em"},
                    ),
                ]),
                dcc.Dropdown(
                    id={"section": section, "field": "__type__",
                        "index": i},
                    options=[
                        {"label": spec.choice.options[k].label, "value": k}
                        for k in spec.choice.keys()
                    ],
                    value=entry["type"],
                    clearable=False,
                ),
                html.Div(
                    form_for_spec(sensor_spec, section, index=i,
                                  values=entry["params"]),
                    id={"section": section, "kind": "entry_form_holder",
                        "index": i},
                    style={"marginTop": "8px"},
                ),
            ],
            style={"border": "1px solid #ccc",
                   "borderRadius": "4px",
                   "padding": "8px",
                   "marginBottom": "8px",
                   "background": "#f9f9f9"},
        ))

    return html.Div(
        [
            html.Label("Sensors", style={"fontWeight": "bold",
                                         "fontSize": "0.95em"}),
            html.Div(cards,
                     id={"section": section, "kind": "list_container",
                         "index": 0}),
            html.Button(
                "+ Add sensor",
                id={"section": section, "field": "__add__", "index": 0},
                n_clicks=0,
                style={"marginTop": "4px"},
            ),
        ],
        style={"marginBottom": "16px"},
    )


def scenario_builder_layout() -> html.Div:
    """
    Full left-sidebar layout: trajectory, motion model, sensor list,
    occlusion, filter, road map, plus simulation controls.
    """
    return html.Div(
        [
            html.H4("Scenario"),
            form_for_choice(TRAJECTORY_CHOICE, section="trajectory"),
            form_for_choice(MOTION_MODEL_CHOICE, section="motion_model"),
            form_for_sensor_list(SENSOR_LIST, section="sensor_list"),
            form_for_choice(OCCLUSION_CHOICE, section="occlusion"),
            form_for_choice(FILTER_CHOICE, section="filter"),
            html.H4("Road map"),
            html.Div([
                dcc.Checklist(
                    id={"section": "road_map", "field": "__enabled__",
                        "index": 0},
                    options=[{"label": " Enable road map (road-aided filtering)",
                              "value": True}],
                    value=[True],
                ),
                form_for_spec(ROAD_MAP, section="road_map"),
            ]),
            html.H4("Simulation"),
            html.Div([
                html.Label("Random seed", style={"fontSize": "0.85em"}),
                dcc.Input(
                    id={"section": "sim", "field": "seed", "index": 0},
                    type="number", value=42, step=1,
                    style={"width": "100%"},
                ),
                html.Label("Time step [s]", style={"fontSize": "0.85em",
                                                  "marginTop": "8px"}),
                dcc.Input(
                    id={"section": "sim", "field": "dt", "index": 0},
                    type="number", value=1.0, min=0.01, max=10.0, step=0.1,
                    style={"width": "100%"},
                ),
            ]),
            html.Button("Run simulation",
                        id="run-simulation",
                        n_clicks=0,
                        style={"marginTop": "16px",
                               "width": "100%",
                               "padding": "8px",
                               "background": "#2b6cb0",
                               "color": "white",
                               "border": "none",
                               "borderRadius": "4px",
                               "cursor": "pointer",
                               "fontSize": "0.95em"}),
        ],
        style={"width": "320px",
               "padding": "12px",
               "background": "#fff",
               "borderRight": "1px solid #ddd",
               "overflowY": "auto",
               "height": "100vh"},
    )
