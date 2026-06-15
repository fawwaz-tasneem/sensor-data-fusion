"""
Simulation runner.

Takes a validated `ScenarioConfig` (a dict of section -> validated values
plus the user's choice of trajectory/filter/etc) and runs an end-to-end
simulation, returning a `SimulationResult` populated with time-series
arrays the playback view consumes.

The runner owns *assembly* — the cross-component wiring that no single
ComponentSpec can express:
  - Build the trajectory.
  - Build the motion model.
  - Build all sensors. Wire occlusion into each. For TunnelOcclusion,
    resolve fractional arc-lengths against the road map. For Doppler,
    inject the sensor's own position.
  - Build the polygonal road map by sampling the trajectory.
  - Build the filter, wiring in the motion model + road map + initial
    state derived from the trajectory's t=0 state plus user prior config.
  - Run the loop, collecting truth, estimates, and per-step metadata.

The runner does *not* draw plots — that's the playback view's job. It
returns raw arrays. Keeping these layers separate means the same
SimulationResult can drive Plotly playback, matplotlib export, or unit
tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from sdf.core.state import StateDistribution
from sdf.filters import (
    ExtendedKalmanFilter,
    IMMFilter,
    KalmanFilter,
    road_cross_track_update,
)
from sdf.motion_models import (
    UnifiedCA,
    UnifiedCT,
    UnifiedCV,
    unified_layout,
)
from sdf.scenarios import (
    FighterJetTrajectory,
    MountainPassTrajectory,
    PolygonalRoadMap,
)
from sdf.sensors import (
    CompositeOcclusion,
    DopplerBlindnessOcclusion,
    TunnelOcclusion,
)


@dataclass
class SimulationResult:
    """All time-series arrays produced by a single simulation run."""

    times: np.ndarray                # (T,)
    truth_positions: np.ndarray      # (T, 3) — always 3D for playback
    estimate_positions: np.ndarray   # (T, 3)
    estimate_pos_std: np.ndarray     # (T, 3) — per-axis 1-sigma for uncertainty bands
    truth_states: np.ndarray         # (T, state_dim) — full state vectors
    estimate_states: np.ndarray      # (T, state_dim)

    # Sensors: positions over time (for moving platforms) and per-step
    # detection flags. Stationary sensors have constant position rows.
    sensor_ids: list[str]
    sensor_positions: np.ndarray     # (T, n_sensors, 3)
    sensor_detected: np.ndarray      # (T, n_sensors) — bool

    # Ground-truth Doppler clutter-notch detection factor per sensor per step,
    # shape (T, n_sensors): 1.0 = clear, ~pd_floor = deep in the notch. NaN for
    # sensors without a Doppler occlusion. This is the *true* notch condition
    # (computed from the truth), independent of the random detection draw, so
    # you can see that misses line up with the notch.
    clutter_factor: Optional[np.ndarray] = None

    # Optional road map for visualization.
    road_nodes: Optional[np.ndarray] = None  # (N, 3) or None

    # Optional tunnel wireframe segments (3D line list).
    tunnel_segments: Optional[list] = None

    # Filter-specific stats. For IMM, this can hold mode probabilities;
    # for v6's other filters it's empty.
    extra: dict = field(default_factory=dict)

    # Non-fatal warnings about the configuration (e.g. a linear KF paired with
    # a nonlinear sensor). Shown to the user, but the run still proceeds.
    warnings: list = field(default_factory=list)

    # IMM mode probabilities over time, shape (T, n_modes), with mode_labels
    # naming the columns (e.g. ["CV", "CT", "CA"]). None for non-IMM filters.
    mode_probs: Optional[np.ndarray] = None
    mode_labels: Optional[list] = None

    # Per-step Euclidean position error ||estimate - truth||, shape (T,).
    position_error: Optional[np.ndarray] = None

    # Scalar evaluation metrics (RMSE, per-axis, consistency, ...). Keyed by
    # name; the playback view renders these in a table.
    metrics: dict = field(default_factory=dict)

    # Echo of the user's config for the playback view to label things.
    config: dict = field(default_factory=dict)


# -- assembly helpers --------------------------------------------------


def _build_trajectory(config: dict):
    """Build the trajectory object from the validated trajectory config."""
    from dashboard.components import TRAJECTORY_CHOICE
    key = config["trajectory"]["type"]
    spec = TRAJECTORY_CHOICE.get(key)
    return spec.construct(config["trajectory"]["params"])


def _build_motion_model(config: dict):
    from dashboard.components import MOTION_MODEL_CHOICE
    key = config["motion_model"]["type"]
    spec = MOTION_MODEL_CHOICE.get(key)
    return spec.construct(config["motion_model"]["params"])


def _build_sensors(config: dict) -> list:
    """Build all sensors. Occlusion wiring is applied separately."""
    from dashboard.components import SENSOR_LIST
    return SENSOR_LIST.build(config["sensor_list"])


def _build_road_map(trajectory, config: dict) -> Optional[PolygonalRoadMap]:
    """
    Build the polygonal road map by sampling the trajectory.

    The road geometry is built for either of two independent reasons:
      * the road map is enabled (so the filter can be road-aided), or
      * a tunnel occlusion is selected (the tunnel is positioned by arc-length
        along the road, so it needs the geometry even when road-aiding is off).
    Building the geometry is separate from *fusing* it — see `road_aided` in
    run_simulation, which gates the fictitious measurement on the checkbox.
    """
    rm_cfg = config.get("road_map", {})
    occ_type = config.get("occlusion", {}).get("type", "none")
    if not rm_cfg.get("enabled", False) and occ_type != "tunnel":
        return None
    n_nodes = rm_cfg["params"]["n_nodes"]
    sigma_nodes = rm_cfg["params"]["sigma_nodes"]

    # Extract the true position at time t via the layout, so this works for any
    # Trajectory (only MountainPassTrajectory exposes a position_at()); every
    # trajectory guarantees state_at() + a layout.
    def _pos(t):
        return trajectory.layout.position(trajectory.state_at(t))

    duration = _trajectory_duration(trajectory)
    times = np.linspace(0.0, duration, n_nodes)
    nodes = np.array([_pos(t) for t in times])

    # Compute true arc length along the smooth curve for each segment.
    arc_lengths = [0.0]
    for i in range(1, n_nodes):
        ts = np.linspace(times[i - 1], times[i], 200)
        pts = np.array([_pos(t) for t in ts])
        seg_arc = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        arc_lengths.append(arc_lengths[-1] + seg_arc)

    return PolygonalRoadMap(
        nodes=nodes,
        arc_lengths=np.array(arc_lengths),
        sigma_nodes=sigma_nodes,
    )


def _trajectory_duration(trajectory) -> float:
    """How long to simulate. Currently derived from trajectory type."""
    if isinstance(trajectory, MountainPassTrajectory):
        return trajectory.length / trajectory.v
    if isinstance(trajectory, FighterJetTrajectory):
        return trajectory.total_duration
    # ConstantVelocityTrajectory: pick a default — 30 seconds at default
    # speed covers most reasonable scenarios. This is a coarse choice
    # that v7 could expose as a top-level "duration" knob.
    return 30.0


def _build_occlusion(config: dict, road_map: Optional[PolygonalRoadMap],
                     sensor_position: np.ndarray):
    """
    Build an occlusion model for a specific sensor.

    Tunnel: shared across sensors (geometric), built once but returned
    each call. Doppler: per-sensor (depends on sensor's own position).
    Both are resolved here against the cross-component context.
    """
    occ_cfg = config.get("occlusion", {})
    key = occ_cfg.get("type", "none")
    if key == "none":
        return None
    if key == "tunnel":
        if road_map is None:
            # Should not happen: _build_road_map builds the geometry whenever a
            # tunnel is selected, independent of the road-aided checkbox.
            raise ValueError(
                "Internal error: tunnel occlusion selected but no road geometry "
                "was built."
            )
        params = occ_cfg["params"]
        total = road_map.total_length()
        return TunnelOcclusion(
            road=road_map,
            l_in=params["l_in_frac"] * total,
            l_out=params["l_out_frac"] * total,
            radius=params["radius"],
        )
    if key == "doppler":
        params = occ_cfg["params"]
        return DopplerBlindnessOcclusion(
            sensor_position=sensor_position,
            mdv=params["mdv"],
            pd_floor=params["pd_floor"],
        )
    raise ValueError(f"unknown occlusion type {key!r}")


def _attach_occlusion(sensors: list, config: dict,
                      road_map: Optional[PolygonalRoadMap]) -> None:
    """Mutate each sensor's `occlusion_model` attribute."""
    occ_cfg = config.get("occlusion", {})
    if occ_cfg.get("type", "none") == "tunnel":
        # One tunnel shared across all sensors.
        shared = _build_occlusion(config, road_map, np.zeros(3))
        for s in sensors:
            s.occlusion_model = shared
    elif occ_cfg.get("type", "none") == "doppler":
        # Doppler is per-sensor — it needs the sensor's own location to compute
        # the clutter notch. A CartesianPositionSensor observes position
        # directly and has NO location, so a Doppler notch is meaningless for
        # it; skip those (they keep occlusion None and always detect).
        for s in sensors:
            pos = getattr(s, "position", None)
            if pos is None:
                continue
            s.occlusion_model = _build_occlusion(config, road_map, pos)
    # else: leave occlusion_model as whatever the sensor was built with
    # (usually None).


def _build_prior(motion_model, trajectory, filter_cfg: dict,
                 rng: np.random.Generator) -> StateDistribution:
    """
    Build the initial StateDistribution for the filter.

    Mean = truth at t=0 + offset (currently fixed direction per axis for
    reproducibility — could randomize via rng if we want noisier priors).
    Covariance = diag([pos_var, vel_var, ...]) tiled per axis in the
    layout-appropriate order.
    """
    truth0 = trajectory.state_at(0.0)
    traj_layout = trajectory.layout
    layout = motion_model.layout

    params = filter_cfg["params"]
    pos_offset = params["prior_position_offset"]
    pos_sigma = params["prior_position_sigma"]
    vel_sigma = params["prior_velocity_sigma"]

    # The filter's state lives in the MOTION MODEL's layout, which may differ
    # from the trajectory's (e.g. a CV trajectory feeding a CA filter, or a
    # 3D trajectory feeding a 2D filter). So we can't just copy the truth
    # vector — we seed the prior by *semantic role*: the filter's position
    # slots get the trajectory's true position, the velocity slots get its
    # true velocity, and any extra slots (acceleration, turn rate) start at
    # zero. We match axes up to the smaller of the two dimensions.
    true_pos = traj_layout.position(truth0)
    true_vel = traj_layout.velocity(truth0)
    mean = np.zeros(motion_model.state_dim)
    for k, idx in enumerate(layout.position_idx):
        if k < len(true_pos):
            mean[idx] = true_pos[k] + pos_offset
    for k, idx in enumerate(layout.velocity_idx):
        if k < len(true_vel):
            mean[idx] = true_vel[k]

    # Build diagonal: position variance at position_idx, velocity variance
    # at velocity_idx, acceleration/turn-rate variance elsewhere (default
    # to 1.0 for any indices we don't know about).
    diag = np.ones(mean.shape[0])
    for idx in layout.position_idx:
        diag[idx] = pos_sigma**2
    for idx in layout.velocity_idx:
        diag[idx] = vel_sigma**2
    if layout.accel_idx is not None:
        for idx in layout.accel_idx:
            diag[idx] = (5.0)**2  # generic accel uncertainty
    if layout.turn_rate_idx is not None:
        diag[layout.turn_rate_idx] = (0.1)**2  # generic turn-rate uncertainty

    return StateDistribution(
        mean=mean,
        covariance=np.diag(diag),
        timestamp=0.0,
        layout=layout,
    )


def _build_filter(motion_model, road_map: Optional[PolygonalRoadMap],
                  trajectory, config: dict, rng: np.random.Generator):
    """Build the configured filter, wiring in motion model + initial state + road map."""
    filter_cfg = config["filter"]
    key = filter_cfg["type"]
    prior = _build_prior(motion_model, trajectory, filter_cfg, rng)

    if key == "kf":
        return KalmanFilter(motion_model=motion_model, initial_state=prior)
    if key == "ekf":
        return ExtendedKalmanFilter(motion_model=motion_model, initial_state=prior)
    if key == "imm":
        # Three-mode IMM on one unified state, sized to the SCENARIO (the
        # trajectory's spatial dimension), NOT the configured motion model — the
        # IMM defines its own bank, so the motion-model dropdown has no effect
        # when IMM is selected. Modes:
        #   mode 0 = CV (smooth, constant velocity)
        #   mode 1 = CT (coordinated turn, adaptive turn rate omega; x-y plane)
        #   mode 2 = CA (constant acceleration, fast maneuvers)
        # 2D state is [x,vx,ax,y,vy,ay,omega]; 3D adds [z,vz,az]. All three
        # share one layout, so the IMM mixes them directly.
        params = filter_cfg["params"]
        dim = trajectory.layout.dim
        layout = unified_layout(dim)
        cv = UnifiedCV(dim=dim, process_noise_std=params["cv_process_noise_std"])
        ct = UnifiedCT(dim=dim, process_noise_std=params["ct_process_noise_std"],
                       omega_noise_std=params["ct_omega_noise_std"])
        ca = UnifiedCA(dim=dim, jerk_std=params["ca_jerk_std"])

        # One shared prior seeded from the trajectory's t=0 truth (position +
        # velocity on every axis); acceleration and turn rate start at zero.
        truth0 = trajectory.state_at(0.0)
        tl = trajectory.layout
        true_pos = tl.position(truth0)
        true_vel = tl.velocity(truth0)
        ps, vs = params["prior_position_sigma"], params["prior_velocity_sigma"]
        mean = np.zeros(3 * dim + 1)
        diag = np.empty(3 * dim + 1)
        for i, idx in enumerate(layout.position_idx):
            mean[idx] = true_pos[i] + params["prior_position_offset"]
            diag[idx] = ps**2
        for i, idx in enumerate(layout.velocity_idx):
            mean[idx] = true_vel[i]
            diag[idx] = vs**2
        for idx in layout.accel_idx:
            diag[idx] = 5.0**2
        diag[layout.turn_rate_idx] = 0.1**2
        prior = StateDistribution(mean=mean, covariance=np.diag(diag),
                                  timestamp=0.0, layout=layout)

        # EKF for every sub-filter: CT is nonlinear, and the EKF update wraps
        # angle innovations correctly for radar-type sensors (a plain KF would
        # not). For the linear CV/CA modes the EKF reduces to the KF.
        sub_filters = [
            ExtendedKalmanFilter(motion_model=cv, initial_state=prior.copy()),
            ExtendedKalmanFilter(motion_model=ct, initial_state=prior.copy()),
            ExtendedKalmanFilter(motion_model=ca, initial_state=prior.copy()),
        ]
        # Transition matrix: tpm_self_prob on the diagonal, rest split.
        p = params["tpm_self_prob"]
        off = (1.0 - p) / 2.0
        tpm = np.array([[p, off, off], [off, p, off], [off, off, p]])
        return IMMFilter(
            filters=sub_filters,
            transition_matrix=tpm,
            mode_probs=np.array([1 / 3, 1 / 3, 1 / 3]),
        )
    raise ValueError(f"unknown filter type {key!r}")


def _check_dimensions(flt, trajectory, sensors, config: dict) -> list[str]:
    """
    Validate scenario compatibility. Hard errors (dimension mismatches) raise;
    soft issues (a linear KF on a nonlinear sensor) are returned as warnings so
    the run still proceeds and its failure is visible.

    The trajectory, the sensors, and the filter's state must share a spatial
    dimension or nothing downstream lines up (cryptic NumPy broadcast errors
    deep inside a filter update). The common trap: coordinated-turn and IMM
    filters are inherently 2D, but the default radar/GMTI sensors and the
    mountain-pass trajectory are 3D — so those mismatches are hard errors.
    """
    spatial = flt.state.layout.dim
    filter_kind = config.get("filter", {}).get("type", "filter")

    if trajectory.layout.dim != spatial:
        raise ValueError(
            f"The {filter_kind!r} filter works in {spatial}D but the trajectory "
            f"is {trajectory.layout.dim}D. Coordinated-turn and IMM filters are "
            f"2D — select the 'Constant velocity' trajectory set to 2D (and 2D "
            f"sensors) to use them."
        )
    for s in sensors:
        sdim = getattr(s, "_dim", spatial)
        if sdim != spatial:
            raise ValueError(
                f"The {filter_kind!r} filter works in {spatial}D but sensor "
                f"{s.sensor_id!r} is {sdim}D. Use a Cartesian sensor set to "
                f"{spatial}D (radar/GMTI are 3D) so the dimensions match."
            )

    # The linear KalmanFilter assumes h(x) = H x. Radar/GMTI/azimuth radars are
    # nonlinear (and their angle outputs live on a circle), so a plain KF will
    # track poorly or diverge — it really wants the EKF. We do NOT block this:
    # running it and seeing the large error is instructive. We just warn.
    warnings: list[str] = []
    if filter_kind == "kf":
        nonlinear = [s.sensor_id for s in sensors if not getattr(s, "is_linear", False)]
        if nonlinear:
            warnings.append(
                "KF (linear) is paired with nonlinear sensor(s): "
                f"{', '.join(nonlinear)}. The KF cannot linearise or wrap angle "
                "innovations, so expect large error / divergence — this is the "
                "expected failure. Switch to EKF for accurate tracking."
            )
    return warnings


def _compute_metrics(
    truth_positions: np.ndarray,
    estimate_positions: np.ndarray,
    estimate_pos_std: np.ndarray,
    truth_states: np.ndarray,
    estimate_states: np.ndarray,
    est_layout,
    traj_layout,
    sensor_detected: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Compute standard tracking-filter evaluation metrics.

    Returns the per-step position-error series and a dict of scalar metrics:
      * RMSE (overall, per-axis, horizontal vs vertical) — the headline
        accuracy numbers. Horizontal vs vertical matters here because an
        azimuth-only radar tracks horizontal well but z poorly.
      * mean / max / final position error.
      * RMSE of velocity (when both states expose velocity).
      * ANEES — Average Normalized Estimation Error Squared, a *consistency*
        check: err^T P^-1 err averaged over time should sit near the number
        of estimated position axes if the filter's reported covariance is
        honest. ANEES >> dim means the filter is over-confident (claims more
        certainty than it has); << dim means it is over-cautious.
      * detection_rate — fraction of steps with at least one detection.

    t = 0 (the prior) is excluded from the averages.
    """
    per_step = np.linalg.norm(estimate_positions - truth_positions, axis=1)
    err = (estimate_positions - truth_positions)[1:]  # drop the prior
    norm_err = np.linalg.norm(err, axis=1)

    def rmse(a: np.ndarray) -> float:
        return float(np.sqrt(np.mean(a**2)))

    metrics: dict = {
        "rmse_position": rmse(norm_err),
        "rmse_x": rmse(err[:, 0]),
        "rmse_y": rmse(err[:, 1]),
        "rmse_z": rmse(err[:, 2]),
        "rmse_horizontal": rmse(np.hypot(err[:, 0], err[:, 1])),
        "rmse_vertical": rmse(err[:, 2]),
        "mean_position_error": float(np.mean(norm_err)),
        "max_position_error": float(np.max(norm_err)),
        "final_position_error": float(per_step[-1]),
    }

    # Velocity RMSE, over whichever spatial axes both layouts share.
    try:
        tv = np.array([traj_layout.velocity(s) for s in truth_states])
        ev = np.array([est_layout.velocity(s) for s in estimate_states])
        d = min(tv.shape[1], ev.shape[1])
        ve = ev[1:, :d] - tv[1:, :d]
        metrics["rmse_velocity"] = rmse(np.linalg.norm(ve, axis=1))
    except Exception:
        pass

    # ANEES consistency, using the per-axis reported std over the estimated
    # position axes (a diagonal approximation — ignores cross-correlations).
    dim = est_layout.dim
    std = estimate_pos_std[1:, :dim]
    erra = (estimate_positions - truth_positions)[1:, :dim]
    with np.errstate(divide="ignore", invalid="ignore"):
        nees = np.sum((erra / std) ** 2, axis=1)
    nees = nees[np.isfinite(nees)]
    if nees.size:
        metrics["anees_position"] = float(np.mean(nees))
        metrics["anees_dof"] = int(dim)

    metrics["detection_rate"] = float(sensor_detected[1:].any(axis=1).mean())
    return per_step, metrics


def _doppler_occlusion(sensor):
    """Return the sensor's DopplerBlindnessOcclusion (direct or inside a
    CompositeOcclusion), or None if it has none."""
    occ = getattr(sensor, "occlusion_model", None)
    if isinstance(occ, DopplerBlindnessOcclusion):
        return occ
    if isinstance(occ, CompositeOcclusion):
        for m in occ.models:
            if isinstance(m, DopplerBlindnessOcclusion):
                return m
    return None


def _pad_to_3d(position: np.ndarray) -> np.ndarray:
    """Lift a 2D position to 3D by appending z=0."""
    if position.shape[-1] == 3:
        return position
    pad = np.zeros((*position.shape[:-1], 1))
    return np.concatenate([position, pad], axis=-1)


# -- the main entry point ---------------------------------------------


def run_simulation(config: dict) -> SimulationResult:
    """
    Run a configured scenario end-to-end.

    `config` must be already validated by the relevant component specs.
    The dashboard's "Run simulation" callback handles that before calling
    here. We assume keys exist; missing keys are programmer errors.
    """
    seed = int(config.get("sim", {}).get("seed", 42))
    dt = float(config.get("sim", {}).get("dt", 1.0))
    rng = np.random.default_rng(seed)

    trajectory = _build_trajectory(config)
    motion_model = _build_motion_model(config)
    road_map = _build_road_map(trajectory, config)
    sensors = _build_sensors(config)
    _attach_occlusion(sensors, config, road_map)
    flt = _build_filter(motion_model, road_map, trajectory, config, rng)

    warnings = _check_dimensions(flt, trajectory, sensors, config)

    # Enabling the road map IS the road-aided switch. But the road map may also
    # be built purely to position a tunnel (occlusion) without aiding — so we
    # gate fusion on the checkbox, not on the geometry's mere existence.
    # Road-aiding is a *fallback*: it is applied only on steps where NO sensor
    # returned a measurement (e.g. inside a tunnel), not on top of live sensor
    # data. With sensors present the sensors do the tracking; the road only
    # takes over to coast the estimate across the gap.
    road_aided = bool(config.get("road_map", {}).get("enabled", False))

    duration = _trajectory_duration(trajectory)
    times = np.arange(0.0, duration + dt, dt)
    T = len(times)

    # Truth and estimate live in their own layouts, which can differ (e.g. an
    # IMM estimate is 2D/4-dim even if the trajectory is 3D). Size each array
    # to its own owner rather than assuming a single shared state dimension.
    traj_layout = trajectory.layout
    est_layout = flt.state.layout
    truth_dim = len(trajectory.state_at(0.0))
    est_dim = flt.state.mean.shape[0]

    truth_states = np.zeros((T, truth_dim))
    estimate_states = np.zeros((T, est_dim))
    truth_positions = np.zeros((T, 3))
    estimate_positions = np.zeros((T, 3))
    estimate_pos_std = np.zeros((T, 3))
    sensor_positions = np.zeros((T, len(sensors), 3))
    sensor_detected = np.zeros((T, len(sensors)), dtype=bool)
    # IMM mode probabilities over time (None for non-IMM filters). The dashboard
    # IMM builds its sub-filters in CV, CT, CA order, so label them that way.
    is_imm = isinstance(flt, IMMFilter)
    mode_probs = np.zeros((T, len(flt.filters))) if is_imm else None
    mode_labels = (["CV", "CT", "CA"][:len(flt.filters)] if is_imm else None)

    def _record_modes(ti: int) -> None:
        if is_imm:
            mode_probs[ti] = flt.mode_probabilities

    # Ground-truth Doppler notch factor; NaN for non-Doppler sensors.
    clutter_factor = np.full((T, len(sensors)), np.nan)
    dopplers = [_doppler_occlusion(s) for s in sensors]
    has_doppler = any(d is not None for d in dopplers)

    def _record_clutter(ti: int) -> None:
        # Call AFTER measure() so a moving GMTI's occlusion pose is synced to t.
        for k, dop in enumerate(dopplers):
            if dop is not None:
                clutter_factor[ti, k] = dop.detection_factor(
                    truth_states[ti], traj_layout
                )

    def _sensor_xyz(s) -> np.ndarray:
        # Cartesian sensors observe position directly and have no location;
        # mark them NaN so the playback can skip drawing a marker for them.
        pos = getattr(s, "position", None)
        return _pad_to_3d(pos) if pos is not None else np.full(3, np.nan)

    def _est_position(state) -> np.ndarray:
        return _pad_to_3d(state.position())

    def _est_pos_std(state) -> np.ndarray:
        return _pad_to_3d(
            np.sqrt(np.array([state.covariance[i, i]
                              for i in est_layout.position_idx]))
        )

    # Initial step.
    truth_states[0] = trajectory.state_at(0.0)
    estimate_states[0] = flt.state.mean
    truth_positions[0] = _pad_to_3d(
        np.array([truth_states[0, i] for i in traj_layout.position_idx])
    )
    estimate_positions[0] = _est_position(flt.state)
    estimate_pos_std[0] = _est_pos_std(flt.state)
    for k, s in enumerate(sensors):
        s.set_time(times[0])
        sensor_positions[0, k] = _sensor_xyz(s)
    _record_clutter(0)
    _record_modes(0)

    for ti in range(1, T):
        t = times[ti]
        x_true = trajectory.state_at(t)
        truth_states[ti] = x_true
        truth_positions[ti] = _pad_to_3d(
            np.array([x_true[i] for i in traj_layout.position_idx])
        )

        flt.predict(t)

        any_detected = False
        for k, s in enumerate(sensors):
            m = s.measure(x_true, traj_layout, t, rng)
            # measure() syncs the sensor's pose to t, so read position after.
            sensor_positions[ti, k] = _sensor_xyz(s)
            if m is not None:
                flt.update(m, s)
                sensor_detected[ti, k] = True
                any_detected = True
        _record_clutter(ti)

        # Road-aided approximation: apply the fictitious cross-track
        # measurement every step the road map is enabled (Koch's standard
        # formulation — the target is always on the road, so the constraint is
        # always valid). This continuously constrains the cross-track subspace,
        # which is what lets it fix the unobservable altitude of an
        # azimuth-only radar and keep the estimate on the road through a tunnel.
        if road_aided:
            if isinstance(flt, IMMFilter):
                # IMM keeps its estimate in per-mode sub-filters; constrain
                # each sub-filter so the constraint survives the next mix.
                for sub in flt.filters:
                    sub.state, _, _ = road_cross_track_update(sub.state, road_map)
                flt._update_combined_state()
            else:
                flt.state, _, _ = road_cross_track_update(flt.state, road_map)

        estimate_states[ti] = flt.state.mean
        estimate_positions[ti] = _est_position(flt.state)
        estimate_pos_std[ti] = _est_pos_std(flt.state)
        _record_modes(ti)

    # Optional tunnel wireframe.
    tunnel_segments = None
    if config.get("occlusion", {}).get("type") == "tunnel" and road_map is not None:
        from sdf.viz import tunnel_wireframe_segments
        # Use the same tunnel we attached to the sensors.
        first_with_tunnel = next(
            (s for s in sensors if isinstance(s.occlusion_model, TunnelOcclusion)),
            None,
        )
        if first_with_tunnel is not None:
            tunnel_segments = tunnel_wireframe_segments(
                first_with_tunnel.occlusion_model,
                n_rings=14, n_long=10,
            )

    position_error, metrics = _compute_metrics(
        truth_positions, estimate_positions, estimate_pos_std,
        truth_states, estimate_states, est_layout, traj_layout, sensor_detected,
    )

    return SimulationResult(
        times=times,
        truth_positions=truth_positions,
        estimate_positions=estimate_positions,
        estimate_pos_std=estimate_pos_std,
        truth_states=truth_states,
        estimate_states=estimate_states,
        sensor_ids=[s.sensor_id for s in sensors],
        sensor_positions=sensor_positions,
        sensor_detected=sensor_detected,
        clutter_factor=clutter_factor if has_doppler else None,
        road_nodes=_pad_to_3d(road_map.nodes) if road_map is not None else None,
        tunnel_segments=tunnel_segments,
        warnings=warnings,
        mode_probs=mode_probs,
        mode_labels=mode_labels,
        position_error=position_error,
        metrics=metrics,
        config=config,
    )
