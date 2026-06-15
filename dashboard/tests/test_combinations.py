"""
Exhaustive dashboard combination test.

Runs (almost) every dimensionally-consistent combination of dashboard options
at DEFAULT parameter values through the real simulation runner, and validates
each result against independently recomputed expectations:

  * shapes are consistent and all estimates/truth are finite;
  * the estimate state dimension matches the chosen filter/motion model;
  * the reported RMSE and detection rate match an independent recomputation;
  * occlusion behaves (none -> always detected, tunnel -> blackouts + wireframe,
    doppler -> clutter factor for located sensors);
  * road-aiding keeps the estimate on the road in a tunnel gap (true
    perpendicular cross-track), and its absence lets it wander off a curved road.

Known-invalid combinations (dimension mismatches) are asserted to raise a clear
ValueError. This is the regression net that stops a fix in one option from
silently breaking another.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from dashboard.components import (
    FILTER_CHOICE,
    MOTION_MODEL_CHOICE,
    OCCLUSION_CHOICE,
    ROAD_MAP,
    SENSOR_CHOICE,
    SENSOR_LIST,
    TRAJECTORY_CHOICE,
)
from dashboard.simulation import (
    _build_road_map,
    _build_trajectory,
    run_simulation,
)

DT = 5.0  # coarse enough to be fast, fine enough that CA does not over-shoot


def _sensor(type_key, **overrides):
    return {"type": type_key,
            "params": SENSOR_CHOICE.get(type_key).validate(overrides)}


def _config(traj, m, sensors, occ, flt, road):
    return {
        "trajectory": traj,
        "motion_model": m,
        "sensor_list": SENSOR_LIST.validate(sensors),
        "occlusion": {"type": occ,
                      "params": OCCLUSION_CHOICE.get(occ).validate({})},
        "filter": {"type": flt, "params": FILTER_CHOICE.get(flt).validate({})},
        "road_map": {"enabled": road, "params": ROAD_MAP.validate({})},
        "sim": {"seed": 7, "dt": DT},
    }


def _traj(type_key, **ov):
    return {"type": type_key,
            "params": TRAJECTORY_CHOICE.get(type_key).validate(ov)}


def _motion(type_key, **ov):
    return {"type": type_key,
            "params": MOTION_MODEL_CHOICE.get(type_key).validate(ov)}


def _expected_est_dim(m_key, m_params, flt, traj_dim):
    # IMM is sized to the scenario (3*dim + 1 unified state), independent of the
    # configured motion model, which it ignores.
    if flt == "imm":
        return 3 * traj_dim + 1
    dim = m_params.get("dim", 2 if m_key.startswith("ct") else 3)
    return {"cv": 2 * dim, "ca": 3 * dim, "ct_known": 4, "ct_unknown": 5}[m_key]


# --- build the case grid -------------------------------------------------

def _cases():
    cases = []

    sensors_3d = {
        "cart3d": [_sensor("cartesian", dim=3)],
        "radar": SENSOR_LIST.defaults(),
        "azimuth": [_sensor("azimuth_radar")],
        "gmti": [_sensor("gmti")],
    }
    for s_name, m_key, flt, occ, road in itertools.product(
        sensors_3d, ["cv", "ca"], ["kf", "ekf", "imm"],
        ["none", "tunnel", "doppler"], [True, False]
    ):
        # IMM ignores the motion model, so only run it once per (sensor, occ,
        # road) rather than for every motion model.
        if flt == "imm" and m_key != "cv":
            continue
        label = f"3D-{s_name}-{m_key}-{flt}-{occ}-road{int(road)}"
        cfg = _config(_traj("mountain_pass"), _motion(m_key, dim=3),
                      sensors_3d[s_name], occ, flt, road)
        cases.append((label, cfg))

    for m_key, flt, occ, road in itertools.product(
        ["cv", "ca", "ct_known", "ct_unknown"], ["kf", "ekf", "imm"],
        ["none", "tunnel", "doppler"], [True, False]
    ):
        label = f"2D-{m_key}-{flt}-{occ}-road{int(road)}"
        m = _motion(m_key, dim=2) if m_key in ("cv", "ca") else _motion(m_key)
        cfg = _config(_traj("constant_velocity", dim=2), m,
                      [_sensor("cartesian", dim=2)], occ, flt, road)
        cases.append((label, cfg))

    for plat in ["stationary", "straight", "circle", "racetrack"]:
        label = f"GMTI-{plat}-doppler"
        cfg = _config(_traj("mountain_pass"), _motion("cv", dim=3),
                      [_sensor("gmti", platform=plat)], "doppler", "ekf", False)
        cases.append((label, cfg))

    # Maneuvering trajectory + IMM, the canonical 3-regime demo, in 2D (with a
    # 2D Cartesian sensor) and 3D (with the default radars).
    cases.append((
        "FIGHTER-2D-imm",
        _config(_traj("fighter_jet", dim=2), _motion("cv", dim=2),
                [_sensor("cartesian", dim=2)], "none", "imm", False),
    ))
    cases.append((
        "FIGHTER-3D-imm-radars",
        _config(_traj("fighter_jet", dim=3), _motion("cv", dim=3),
                SENSOR_LIST.defaults(), "none", "imm", False),
    ))

    return cases


CASES = _cases()


@pytest.mark.parametrize("label,cfg", CASES, ids=[c[0] for c in CASES])
def test_combination(label, cfg):
    r = run_simulation(cfg)
    T = len(r.times)

    # shapes + finiteness
    assert r.truth_positions.shape == (T, 3)
    assert r.estimate_positions.shape == (T, 3)
    assert np.isfinite(r.truth_positions).all()
    assert np.isfinite(r.estimate_positions).all()
    assert np.isfinite(r.estimate_states).all()

    # estimate state dimension
    exp = _expected_est_dim(cfg["motion_model"]["type"],
                            cfg["motion_model"]["params"], cfg["filter"]["type"],
                            _build_trajectory(cfg).layout.dim)
    assert r.estimate_states.shape[1] == exp

    # metrics recomputed independently
    err = np.linalg.norm(r.estimate_positions - r.truth_positions, axis=1)[1:]
    assert abs(r.metrics["rmse_position"] - np.sqrt((err ** 2).mean())) < 1e-6
    det_indep = float(r.sensor_detected[1:].any(axis=1).mean())
    assert abs(r.metrics["detection_rate"] - det_indep) < 1e-9

    occ = cfg["occlusion"]["type"]
    if occ == "none":
        assert r.sensor_detected[1:].any(axis=1).all()  # P_D=1, no occlusion
    if occ == "tunnel":
        assert (~r.sensor_detected.any(axis=1)).sum() > 0
        assert r.tunnel_segments is not None
    if occ == "doppler":
        sens = SENSOR_LIST.build(cfg["sensor_list"])
        if any(getattr(s, "position", None) is not None for s in sens):
            assert r.clutter_factor is not None

    # road-aiding physics through a tunnel gap (true perpendicular cross-track)
    if occ == "tunnel":
        rm = _build_road_map(_build_trajectory(cfg), cfg)
        gap = ~r.sensor_detected.any(axis=1)
        if rm is not None and gap.any():
            perp = []
            for p in r.estimate_positions[gap]:
                seg, foot, _ = rm.closest_segment(p[:rm.dim])
                N = rm.cross_track_normals(seg)
                perp.append(float(np.linalg.norm(N @ (p[:rm.dim] - foot))))
            mx = max(perp)
            curved = cfg["trajectory"]["type"] == "mountain_pass"
            if cfg["road_map"]["enabled"]:
                assert mx < 300.0, f"{label}: road ON but perp cross-track {mx:.0f}m"
            elif curved:
                assert mx > 100.0, f"{label}: road OFF (curved) but hugged road {mx:.0f}m"


@pytest.mark.parametrize("label,cfg", [
    # A coordinated-turn MOTION model is 2D-only, so it can't run with 3D radars.
    # (The IMM, by contrast, now works in 3D, so it is no longer invalid here.)
    ("ct-needs-2D", _config(_traj("mountain_pass"), _motion("ct_known"),
                            SENSOR_LIST.defaults(), "none", "ekf", False)),
])
def test_invalid_combination_raises_clear_error(label, cfg):
    with pytest.raises(ValueError):
        run_simulation(cfg)


def test_gmti_platforms_move():
    """Each non-stationary GMTI flight pattern must actually move."""
    for plat in ["straight", "circle", "racetrack"]:
        cfg = _config(_traj("mountain_pass"), _motion("cv", dim=3),
                      [_sensor("gmti", platform=plat)], "doppler", "ekf", False)
        r = run_simulation(cfg)
        moved = np.linalg.norm(r.sensor_positions[-1, 0] - r.sensor_positions[0, 0])
        assert moved > 100.0, f"GMTI {plat} barely moved ({moved:.0f}m)"
