"""
Main Dash app for the SDF scenario builder + playback dashboard.

Layout: left sidebar (scenario_builder_layout) + right pane (initially
empty, populated with playback_layout when a run completes).

Callback architecture:
  * `Store("sensor-list-store")` holds the list of sensor entries.
    Add/remove buttons mutate it; the rendered form below it follows.
  * `Store("result-store")` holds the latest SimulationResult.
  * "Run simulation" button reads all visible form values + sensor list,
    assembles a config dict, runs the simulation, writes the result.
  * "Export MP4" button reads the result store, renders an MP4, and
    offers it as a download.

Forms re-render when their choice dropdown changes (e.g. switching
trajectory type). We do this by re-running the form generator and
replacing the holder div's children.
"""
from __future__ import annotations

import base64
import io
import logging
import traceback
from pathlib import Path
from typing import Any

from dash import (
    ALL,
    Dash,
    Input,
    Output,
    State,
    callback_context,
    ctx,
    dcc,
    html,
    no_update,
)

from dashboard.components import (
    FILTER_CHOICE,
    MOTION_MODEL_CHOICE,
    OCCLUSION_CHOICE,
    ROAD_MAP,
    SENSOR_LIST,
    TRAJECTORY_CHOICE,
)
from dashboard.mp4_export import export_mp4_to_tempfile
from dashboard.schema import ComponentChoice
from dashboard.simulation import SimulationResult, run_simulation
from dashboard.ui import form_for_choice, form_for_sensor_list, form_for_spec
from dashboard.ui.forms import scenario_builder_layout
from dashboard.ui.playback import playback_layout

log = logging.getLogger(__name__)


# Cache the latest SimulationResult on the server side. Dash Store
# objects can't hold arbitrary Python objects (only JSON), so we keep
# results in a module-level dict keyed by a session-id-ish counter.
# This is simple and works for a single-user development dashboard;
# multi-user deployment would need a real server-side cache.
_RESULTS: dict[int, SimulationResult] = {}
_NEXT_RESULT_ID = [0]


def _store_result(result: SimulationResult) -> int:
    """Store a result and return its id (a tiny integer that goes in the JSON store)."""
    rid = _NEXT_RESULT_ID[0]
    _NEXT_RESULT_ID[0] += 1
    _RESULTS[rid] = result
    # Keep only the last 5 results to bound memory.
    if len(_RESULTS) > 5:
        oldest = min(_RESULTS.keys())
        del _RESULTS[oldest]
    return rid


def _fetch_result(rid: int) -> SimulationResult | None:
    return _RESULTS.get(rid)


# ----------------------------------------------------------------------
# Helpers that turn the form's pattern-matching values into a config dict
# ----------------------------------------------------------------------


def _gather_section_params(
    field_ids: list[dict],
    values: list[Any],
    section: str,
) -> dict[str, Any]:
    """
    Collect all form values for a given `section` into a {field_name: value}
    dict, reconstructing vector-slot fields into lists.

    `field_ids` and `values` come from Dash's `State(ALL, ...)` bindings.
    """
    out: dict[str, Any] = {}
    vector_slots: dict[str, dict[int, Any]] = {}

    for fid, val in zip(field_ids, values):
        if fid is None or fid.get("section") != section:
            continue
        field = fid.get("field")
        if field is None or field.startswith("__"):
            continue
        if "slot" in fid:
            vector_slots.setdefault(field, {})[fid["slot"]] = val
        else:
            out[field] = val

    for field, slots in vector_slots.items():
        out[field] = [slots[k] for k in sorted(slots)]

    return out


# ----------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------


def create_app() -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "SDF Scenario Builder"

    app.layout = html.Div(
        [
            # Server-side caches.
            dcc.Store(id="sensor-list-store",
                      data=SENSOR_LIST.defaults()),
            dcc.Store(id="result-id-store", data=None),
            dcc.Store(id="error-store", data=None),
            # Download component for MP4 export.
            dcc.Download(id="mp4-download"),

            html.Div(
                [
                    # Left sidebar with all the controls.
                    html.Div(
                        scenario_builder_layout(),
                        id="sidebar-container",
                        style={"flexShrink": "0"},
                    ),
                    # Right pane: status + playback.
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Playback",
                                            style={"display": "inline-block",
                                                   "marginRight": "20px"}),
                                    html.Button(
                                        "Export MP4",
                                        id="export-mp4-btn",
                                        n_clicks=0,
                                        style={"padding": "6px 14px",
                                               "background": "#38a169",
                                               "color": "white",
                                               "border": "none",
                                               "borderRadius": "4px",
                                               "cursor": "pointer"},
                                    ),
                                    html.Span(id="status-line",
                                              style={"marginLeft": "20px",
                                                     "color": "#666",
                                                     "fontSize": "0.9em"}),
                                ],
                                style={"padding": "12px 20px 0 20px"},
                            ),
                            html.Div(
                                playback_layout(None),
                                id="playback-container",
                                style={"padding": "12px 20px"},
                            ),
                        ],
                        style={"flex": "1", "overflow": "auto"},
                    ),
                ],
                style={"display": "flex", "flexDirection": "row",
                       "height": "100vh", "background": "#f5f5f5"},
            ),
        ],
    )

    _register_callbacks(app)
    return app


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------


def _register_callbacks(app: Dash) -> None:

    # --- Re-render a single-choice form when its dropdown changes ----
    def _choice_form_callback(choice: ComponentChoice, section: str):
        @app.callback(
            Output({"section": section, "kind": "choice_form_holder",
                    "index": 0}, "children"),
            Input({"section": section, "field": "__choice__",
                   "index": 0}, "value"),
        )
        def update_form(selected_key):
            spec = choice.get(selected_key)
            return form_for_spec(spec, section=section, index=0)

    _choice_form_callback(TRAJECTORY_CHOICE, "trajectory")
    _choice_form_callback(MOTION_MODEL_CHOICE, "motion_model")
    _choice_form_callback(OCCLUSION_CHOICE, "occlusion")
    _choice_form_callback(FILTER_CHOICE, "filter")

    # --- Sensor list: add/remove entries ----------------------------
    @app.callback(
        Output("sensor-list-store", "data"),
        Input({"section": "sensor_list", "field": "__add__",
               "index": 0}, "n_clicks"),
        Input({"section": "sensor_list", "field": "__remove__",
               "index": ALL}, "n_clicks"),
        State("sensor-list-store", "data"),
        prevent_initial_call=True,
    )
    def mutate_sensor_list(add_clicks, remove_clicks, current):
        triggered = ctx.triggered_id
        if triggered is None:
            return no_update
        # Pattern-matching callbacks fire on initial load when new IDs
        # matching the pattern appear, even with prevent_initial_call.
        # Guard by checking that the triggering button was actually clicked.
        triggered_prop = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
        triggered_value = ctx.triggered[0]["value"] if ctx.triggered else None
        if not triggered_value:
            # n_clicks was 0/None — this is the spurious initial firing.
            return no_update
        # Make a working copy.
        entries = list(current) if current else []
        if triggered == {"section": "sensor_list", "field": "__add__",
                         "index": 0}:
            # Add a fresh default sensor (use the choice's default key).
            default_key = SENSOR_LIST.choice.default_key
            default_spec = SENSOR_LIST.choice.get(default_key)
            entries.append({
                "type": default_key,
                "params": default_spec.defaults(),
            })
        elif isinstance(triggered, dict) and triggered.get("field") == "__remove__":
            i = triggered["index"]
            if 0 <= i < len(entries):
                entries.pop(i)
        return entries

    # --- Re-render the sensor list when the store changes ----------
    @app.callback(
        Output({"section": "sensor_list", "kind": "list_container",
                "index": 0}, "children"),
        Input("sensor-list-store", "data"),
    )
    def render_sensor_list(entries):
        if entries is None:
            entries = SENSOR_LIST.defaults()
        # We re-use form_for_sensor_list but extract only the inner
        # container's children to avoid double-wrapping.
        full = form_for_sensor_list(SENSOR_LIST, section="sensor_list",
                                    entries=entries)
        # form_for_sensor_list returns: Div[Label, Div(container), Button].
        # We want the children of the container Div (index 1 in the outer Div).
        container = full.children[1]
        return container.children

    # --- Re-render a per-sensor form when its type dropdown changes -
    @app.callback(
        Output({"section": "sensor_list", "kind": "entry_form_holder",
                "index": ALL}, "children"),
        Input({"section": "sensor_list", "field": "__type__",
               "index": ALL}, "value"),
        State("sensor-list-store", "data"),
    )
    def update_sensor_forms(type_values, entries):
        if not type_values or not entries:
            return [no_update for _ in (type_values or [])]
        out = []
        for i, t in enumerate(type_values):
            if t is None:
                out.append(no_update)
                continue
            spec = SENSOR_LIST.choice.get(t)
            # If the type matches the stored entry, use its params;
            # otherwise default to the new spec's defaults.
            if i < len(entries) and entries[i].get("type") == t:
                params = entries[i]["params"]
            else:
                params = spec.defaults()
            out.append(form_for_spec(spec, section="sensor_list",
                                     index=i, values=params))
        return out

    # --- Run simulation -------------------------------------------
    @app.callback(
        Output("result-id-store", "data"),
        Output("error-store", "data"),
        Output("status-line", "children"),
        Input("run-simulation", "n_clicks"),
        # Choice keys for each section.
        State({"section": "trajectory", "field": "__choice__",
               "index": 0}, "value"),
        State({"section": "motion_model", "field": "__choice__",
               "index": 0}, "value"),
        State({"section": "occlusion", "field": "__choice__",
               "index": 0}, "value"),
        State({"section": "filter", "field": "__choice__",
               "index": 0}, "value"),
        # All form fields (pattern-matching).
        State({"section": ALL, "field": ALL, "index": ALL}, "id"),
        State({"section": ALL, "field": ALL, "index": ALL}, "value"),
        State({"section": ALL, "field": ALL, "index": ALL,
               "slot": ALL}, "id"),
        State({"section": ALL, "field": ALL, "index": ALL,
               "slot": ALL}, "value"),
        # Road map enabled checkbox.
        State({"section": "road_map", "field": "__enabled__",
               "index": 0}, "value"),
        # Sensor list.
        State("sensor-list-store", "data"),
        prevent_initial_call=True,
    )
    def do_run(n, traj_key, mm_key, occ_key, flt_key,
               scalar_ids, scalar_vals,
               vector_ids, vector_vals,
               road_enabled_value,
               sensor_entries):
        try:
            all_ids = (scalar_ids or []) + (vector_ids or [])
            all_vals = (scalar_vals or []) + (vector_vals or [])

            traj_params = _gather_section_params(all_ids, all_vals,
                                                 "trajectory")
            mm_params = _gather_section_params(all_ids, all_vals,
                                               "motion_model")
            occ_params = _gather_section_params(all_ids, all_vals,
                                                "occlusion")
            flt_params = _gather_section_params(all_ids, all_vals,
                                                "filter")
            rm_params = _gather_section_params(all_ids, all_vals,
                                               "road_map")
            sim_params = _gather_section_params(all_ids, all_vals,
                                                "sim")
            sensor_list = (sensor_entries
                           if sensor_entries is not None
                           else SENSOR_LIST.defaults())

            # Validate per-section via specs.
            road_enabled = bool(road_enabled_value)
            config = {
                "trajectory": {
                    "type": traj_key,
                    "params": TRAJECTORY_CHOICE.get(traj_key)
                                .validate(traj_params),
                },
                "motion_model": {
                    "type": mm_key,
                    "params": MOTION_MODEL_CHOICE.get(mm_key)
                                .validate(mm_params),
                },
                "sensor_list": SENSOR_LIST.validate(sensor_list),
                "occlusion": {
                    "type": occ_key,
                    "params": OCCLUSION_CHOICE.get(occ_key)
                                .validate(occ_params),
                },
                "filter": {
                    "type": flt_key,
                    "params": FILTER_CHOICE.get(flt_key)
                                .validate(flt_params),
                },
                "road_map": {
                    "enabled": road_enabled,
                    "params": ROAD_MAP.validate(rm_params),
                },
                "sim": {
                    "seed": int(sim_params.get("seed", 42)),
                    "dt": float(sim_params.get("dt", 1.0)),
                },
            }

            result = run_simulation(config)
            rid = _store_result(result)
            status = (
                f"Run #{rid}: T={len(result.times)} steps, "
                f"{len(result.sensor_ids)} sensors"
            )
            return rid, None, status
        except Exception as exc:
            log.exception("Simulation failed")
            err = f"Simulation failed: {exc}"
            return no_update, err, err

    # --- Update playback when result-id-store changes ---------------
    @app.callback(
        Output("playback-container", "children"),
        Input("result-id-store", "data"),
    )
    def render_playback(rid):
        if rid is None:
            return playback_layout(None)
        result = _fetch_result(rid)
        return playback_layout(result)

    # --- MP4 export -------------------------------------------------
    @app.callback(
        Output("mp4-download", "data"),
        Output("status-line", "children", allow_duplicate=True),
        Input("export-mp4-btn", "n_clicks"),
        State("result-id-store", "data"),
        prevent_initial_call=True,
    )
    def export_mp4(n, rid):
        if rid is None:
            return no_update, "Nothing to export — run a simulation first."
        result = _fetch_result(rid)
        if result is None:
            return no_update, "Result expired from cache — re-run."
        try:
            path = export_mp4_to_tempfile(result)
            return dcc.send_file(str(path), filename=f"sdf_run_{rid}.mp4"), \
                   f"Exported {path.name}"
        except RuntimeError as exc:
            return no_update, str(exc)
        except Exception as exc:
            log.exception("MP4 export failed")
            return no_update, f"MP4 export failed: {exc}"


# ----------------------------------------------------------------------
# Module-level instance for `python -m dashboard`
# ----------------------------------------------------------------------


app = create_app()
server = app.server  # for gunicorn / production deploy
