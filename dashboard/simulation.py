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

from sdf.core.state import StateDistribution, StateLayout
from sdf.filters import (
    ExtendedKalmanFilter,
    IMMFilter,
    KalmanFilter,
    RoadAidedExtendedKalmanFilter,
)
from sdf.motion_models import (
    ConstantAcceleration,
    ConstantVelocity,
    CoordinatedTurn,
)
from sdf.scenarios import (
    MountainPassTrajectory,
    PolygonalRoadMap,
)
from sdf.sensors import (
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

    # Optional road map for visualization.
    road_nodes: Optional[np.ndarray] = None  # (N, 3) or None

    # Optional tunnel wireframe segments (3D line list).
    tunnel_segments: Optional[list] = None

    # Filter-specific stats. For IMM, this can hold mode probabilities;
    # for v6's other filters it's empty.
    extra: dict = field(default_factory=dict)

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
    """Build the polygonal road map by sampling the trajectory."""
    rm_cfg = config.get("road_map", {})
    if not rm_cfg.get("enabled", False):
        return None
    n_nodes = rm_cfg["params"]["n_nodes"]
    sigma_nodes = rm_cfg["params"]["sigma_nodes"]

    duration = _trajectory_duration(trajectory)
    times = np.linspace(0.0, duration, n_nodes)
    nodes = np.array([trajectory.position_at(t) for t in times])

    # Compute true arc length along the smooth curve for each segment.
    arc_lengths = [0.0]
    for i in range(1, n_nodes):
        ts = np.linspace(times[i - 1], times[i], 200)
        pts = np.array([trajectory.position_at(t) for t in ts])
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
            # User asked for a tunnel but didn't enable the road map —
            # silently disable rather than crash. The dashboard should
            # ideally prevent this at the UI level, but the runner needs
            # to be defensive.
            return None
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
        # Doppler is per-sensor — re-built for each sensor's position.
        for s in sensors:
            s.occlusion_model = _build_occlusion(config, road_map, s.position)
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
    layout = motion_model.layout

    params = filter_cfg["params"]
    pos_offset = params["prior_position_offset"]
    pos_sigma = params["prior_position_sigma"]
    vel_sigma = params["prior_velocity_sigma"]

    mean = truth0.copy()
    for idx in layout.position_idx:
        mean[idx] += pos_offset

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
    if key == "road_aided_ekf":
        if road_map is None:
            raise ValueError(
                "Road-aided EKF requires the road map to be enabled."
            )
        return RoadAidedExtendedKalmanFilter(
            motion_model=motion_model, initial_state=prior, road_map=road_map,
        )
    if key == "imm":
        # IMM with fixed sub-models: CV + CA + CT-known.
        params = filter_cfg["params"]
        # All sub-models share the layout of the *configured* motion model.
        # In v6 we constrain IMM to 2D state space (CT is 2D-only). If the
        # user has 3D selected elsewhere, fall back to 2D for IMM sub-models.
        dim = 2
        sub_cv = ConstantVelocity(dim=dim,
                                  process_noise_std=params["cv_process_noise_std"])
        sub_ca = ConstantAcceleration(dim=dim,
                                      jerk_std=params["ca_jerk_std"])
        sub_ct = CoordinatedTurn(omega=params["ct_omega"],
                                 process_noise_std=params["ct_process_noise_std"])
        # IMM needs per-sub-model priors with matching state dimensions.
        sub_filters = []
        for sm in (sub_cv, sub_ca, sub_ct):
            sub_layout = sm.layout
            sub_dim_state = sub_layout.dim_state if hasattr(sub_layout, "dim_state") else sm.transition_matrix(1.0).shape[0]
            sub_mean = np.zeros(sub_dim_state)
            # Copy what we can from the configured prior's position/velocity.
            truth0 = trajectory.state_at(0.0)
            for k, idx in enumerate(sub_layout.position_idx):
                if k < len(motion_model.layout.position_idx):
                    sub_mean[idx] = truth0[motion_model.layout.position_idx[k]] + params["prior_position_offset"]
            for k, idx in enumerate(sub_layout.velocity_idx):
                if k < len(motion_model.layout.velocity_idx):
                    sub_mean[idx] = truth0[motion_model.layout.velocity_idx[k]]
            sub_cov = np.diag([params["prior_position_sigma"]**2 if i in sub_layout.position_idx else
                               params["prior_velocity_sigma"]**2 if i in sub_layout.velocity_idx else
                               5.0**2 for i in range(sub_dim_state)])
            sub_filters.append(ExtendedKalmanFilter(
                motion_model=sm,
                initial_state=StateDistribution(
                    mean=sub_mean, covariance=sub_cov, timestamp=0.0,
                    layout=sub_layout,
                ),
            ))
        # Transition matrix: tpm_self_prob on the diagonal, rest split.
        p = params["tpm_self_prob"]
        off = (1.0 - p) / 2.0
        tpm = np.array([[p, off, off], [off, p, off], [off, off, p]])
        return IMMFilter(
            filters=sub_filters,
            transition_matrix=tpm,
            mode_probs=np.array([1/3, 1/3, 1/3]),
        )
    raise ValueError(f"unknown filter type {key!r}")


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

    duration = _trajectory_duration(trajectory)
    times = np.arange(0.0, duration + dt, dt)
    T = len(times)
    state_dim = motion_model.layout.dim_state if hasattr(motion_model.layout, "dim_state") else len(trajectory.state_at(0.0))

    truth_states = np.zeros((T, state_dim))
    estimate_states = np.zeros((T, state_dim))
    truth_positions = np.zeros((T, 3))
    estimate_positions = np.zeros((T, 3))
    estimate_pos_std = np.zeros((T, 3))
    sensor_positions = np.zeros((T, len(sensors), 3))
    sensor_detected = np.zeros((T, len(sensors)), dtype=bool)

    layout = motion_model.layout
    traj_layout = trajectory.layout

    # Initial step.
    truth_states[0] = trajectory.state_at(0.0)
    estimate_states[0] = flt.state.mean
    truth_positions[0] = _pad_to_3d(
        np.array([truth_states[0, i] for i in traj_layout.position_idx])
    )
    estimate_positions[0] = _pad_to_3d(
        np.array([estimate_states[0, i] for i in layout.position_idx])
    )
    estimate_pos_std[0] = _pad_to_3d(
        np.sqrt(np.array([flt.state.covariance[i, i] for i in layout.position_idx]))
    )
    for k, s in enumerate(sensors):
        sensor_positions[0, k] = _pad_to_3d(s.position)

    for ti in range(1, T):
        t = times[ti]
        x_true = trajectory.state_at(t)
        truth_states[ti] = x_true
        truth_positions[ti] = _pad_to_3d(
            np.array([x_true[i] for i in traj_layout.position_idx])
        )

        flt.predict(t)

        for k, s in enumerate(sensors):
            sensor_positions[ti, k] = _pad_to_3d(s.position)
            m = s.measure(x_true, traj_layout, t, rng)
            if m is not None:
                flt.update(m, s)
                sensor_detected[ti, k] = True

        if isinstance(flt, RoadAidedExtendedKalmanFilter):
            flt.update_with_road()

        estimate_states[ti] = flt.state.mean
        estimate_positions[ti] = _pad_to_3d(
            np.array([flt.state.mean[i] for i in layout.position_idx])
        )
        estimate_pos_std[ti] = _pad_to_3d(
            np.sqrt(np.array([flt.state.covariance[i, i] for i in layout.position_idx]))
        )

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
        road_nodes=_pad_to_3d(road_map.nodes) if road_map is not None else None,
        tunnel_segments=tunnel_segments,
        config=config,
    )
