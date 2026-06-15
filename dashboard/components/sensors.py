"""
Sensor ComponentSpecs.

Each sensor has its own parameter form. RadarSensor and GMTIRadarSensor
take a 3D position as `np.ndarray`, so we use a custom `build` to convert
from form list to ndarray. Occlusion model is wired in by the simulation
runner, not here, because it's a cross-component dependency.
"""
from __future__ import annotations

import numpy as np

from sdf.scenarios.awacs import CircleFlight, RacetrackFlight, StraightFlight
from sdf.sensors import (
    AzimuthOnlyRadarSensor,
    CartesianPositionSensor,
    GMTIRadarSensor,
    RadarSensor,
)

from dashboard.schema import ComponentChoice, ComponentSpec, ParameterSpec


def _build_platform(values: dict):
    """
    Assemble the GMTI's carrier platform from the flat sensor form.

    The sensor's `position` doubles as the platform anchor: the start point of
    a straight flight, or the centre of a circle / racetrack orbit. The other
    motion parameters (velocity, speed, radius, leg length) only apply to some
    patterns; the rest are ignored. Returns None for a stationary platform so
    the GMTI just sits at `position`.
    """
    kind = values.get("platform", "stationary")
    anchor = np.asarray(values["position"], dtype=float)
    if kind == "stationary":
        return None
    if kind == "straight":
        return StraightFlight(
            start_position=anchor,
            velocity=np.asarray(values["platform_velocity"], dtype=float),
        )
    if kind == "circle":
        return CircleFlight(
            center=anchor,
            radius=values["platform_radius"],
            speed=values["platform_speed"],
        )
    if kind == "racetrack":
        return RacetrackFlight(
            center=anchor,
            leg_length=values["platform_leg_length"],
            radius=values["platform_radius"],
            speed=values["platform_speed"],
        )
    raise ValueError(f"unknown GMTI platform {kind!r}")


def _build_radar(values: dict) -> RadarSensor:
    return RadarSensor(
        sensor_id=values["sensor_id"],
        position=np.asarray(values["position"], dtype=float),
        range_std=values["range_std"],
        bearing_std=values["bearing_std"],
        elevation_std=values["elevation_std"],
        detection_prob=values["detection_prob"],
    )


def _build_azimuth_radar(values: dict) -> AzimuthOnlyRadarSensor:
    return AzimuthOnlyRadarSensor(
        sensor_id=values["sensor_id"],
        position=np.asarray(values["position"], dtype=float),
        range_std=values["range_std"],
        bearing_std=values["bearing_std"],
        detection_prob=values["detection_prob"],
    )


def _build_gmti(values: dict) -> GMTIRadarSensor:
    return GMTIRadarSensor(
        sensor_id=values["sensor_id"],
        position=np.asarray(values["position"], dtype=float),
        range_std=values["range_std"],
        bearing_std=values["bearing_std"],
        elevation_std=values["elevation_std"],
        range_rate_std=values["range_rate_std"],
        detection_prob=values["detection_prob"],
        platform=_build_platform(values),
    )


CARTESIAN = ComponentSpec(
    label="Cartesian position sensor",
    constructor=CartesianPositionSensor,
    description="Direct position measurement with isotropic Gaussian noise.",
    parameters=[
        ParameterSpec("sensor_id", str, default="cart_1",
                      description="Unique sensor identifier"),
        ParameterSpec("dim", int, default=3,
                      choices=[("2D", 2), ("3D", 3)],
                      description="Measurement dimension"),
        ParameterSpec("noise_std", float, default=4.0,
                      min=0.1, max=100.0, step=0.1,
                      description="Per-axis noise std-dev", unit="m"),
        ParameterSpec("detection_prob", float, default=1.0,
                      min=0.0, max=1.0, step=0.01,
                      description="Probability of detection"),
    ],
)


RADAR = ComponentSpec(
    label="Radar (range / bearing / elevation)",
    constructor=RadarSensor,
    build=_build_radar,
    description="Stationary radar with nonlinear range/azimuth/elevation measurements.",
    parameters=[
        ParameterSpec("sensor_id", str, default="radar_a",
                      description="Unique sensor identifier"),
        ParameterSpec("position", float, default=[0.0, 10_000.0, 100.0], length=3,
                      description="Sensor position (x, y, z)", unit="m"),
        ParameterSpec("range_std", float, default=80.0,
                      min=1.0, max=500.0, step=1.0,
                      description="Range noise std-dev", unit="m"),
        ParameterSpec("bearing_std", float, default=8e-3,
                      min=1e-4, max=0.1, step=1e-4,
                      description="Bearing noise std-dev", unit="rad"),
        ParameterSpec("elevation_std", float, default=8e-3,
                      min=1e-4, max=0.1, step=1e-4,
                      description="Elevation noise std-dev", unit="rad"),
        ParameterSpec("detection_prob", float, default=1.0,
                      min=0.0, max=1.0, step=0.01,
                      description="Probability of detection"),
    ],
)


GMTI = ComponentSpec(
    label="GMTI radar (range + range-rate)",
    constructor=GMTIRadarSensor,
    build=_build_gmti,
    description=(
        "Ground moving target indicator radar; reports range-rate in addition "
        "to range/azimuth/elevation. Can ride a moving platform (a low-flying "
        "plane) — pick a flight pattern below. Pair it with a Doppler clutter "
        "notch occlusion to model targets lost in clutter."
    ),
    parameters=[
        ParameterSpec("sensor_id", str, default="gmti_a",
                      description="Unique sensor identifier"),
        ParameterSpec("position", float, default=[0.0, 10_000.0, 100.0], length=3,
                      description="Sensor position / platform anchor (x, y, z)",
                      unit="m"),
        ParameterSpec("range_std", float, default=80.0,
                      min=1.0, max=500.0, step=1.0,
                      description="Range noise std-dev", unit="m"),
        ParameterSpec("bearing_std", float, default=8e-3,
                      min=1e-4, max=0.1, step=1e-4,
                      description="Bearing noise std-dev", unit="rad"),
        ParameterSpec("elevation_std", float, default=8e-3,
                      min=1e-4, max=0.1, step=1e-4,
                      description="Elevation noise std-dev", unit="rad"),
        ParameterSpec("range_rate_std", float, default=0.5,
                      min=0.01, max=10.0, step=0.01,
                      description="Range-rate noise std-dev", unit="m/s"),
        ParameterSpec("detection_prob", float, default=1.0,
                      min=0.0, max=1.0, step=0.01,
                      description="Probability of detection"),
        # --- Platform (flight pattern of a moving GMTI) ----------------
        ParameterSpec("platform", str, default="stationary",
                      choices=[("Stationary", "stationary"),
                               ("Straight flight", "straight"),
                               ("Circle", "circle"),
                               ("Racetrack (oval)", "racetrack")],
                      description="Platform / flight pattern"),
        ParameterSpec("platform_velocity", float,
                      default=[150.0, 0.0, 0.0], length=3,
                      description="Velocity for straight flight (vx, vy, vz)",
                      unit="m/s"),
        ParameterSpec("platform_speed", float, default=150.0,
                      min=10.0, max=500.0, step=1.0,
                      description="Ground speed for circle / racetrack",
                      unit="m/s"),
        ParameterSpec("platform_radius", float, default=5_000.0,
                      min=200.0, max=100_000.0, step=100.0,
                      description="Orbit / turn radius for circle / racetrack",
                      unit="m"),
        ParameterSpec("platform_leg_length", float, default=10_000.0,
                      min=500.0, max=200_000.0, step=100.0,
                      description="Straight-leg length for racetrack", unit="m"),
    ],
)


AZIMUTH_RADAR = ComponentSpec(
    label="Azimuth-only radar (range + bearing, no elevation)",
    constructor=AzimuthOnlyRadarSensor,
    build=_build_azimuth_radar,
    description=(
        "3D-sited radar with no height-finding: measures range and azimuth "
        "only. Azimuth carries no vertical information, so the target's "
        "altitude is effectively unobservable from this sensor alone — pair "
        "it with the road map to constrain z. This is the classic 2D "
        "surveillance-radar / sensor-simulator model."
    ),
    parameters=[
        ParameterSpec("sensor_id", str, default="azradar_a",
                      description="Unique sensor identifier"),
        ParameterSpec("position", float, default=[0.0, 10_000.0, 100.0], length=3,
                      description="Sensor position (x, y, z)", unit="m"),
        ParameterSpec("range_std", float, default=80.0,
                      min=1.0, max=500.0, step=1.0,
                      description="Range noise std-dev", unit="m"),
        ParameterSpec("bearing_std", float, default=8e-3,
                      min=1e-4, max=0.1, step=1e-4,
                      description="Azimuth noise std-dev", unit="rad"),
        ParameterSpec("detection_prob", float, default=1.0,
                      min=0.0, max=1.0, step=0.01,
                      description="Probability of detection"),
    ],
)


SENSOR_CHOICE = ComponentChoice(
    label="Sensor",
    options={
        "cartesian": CARTESIAN,
        "radar": RADAR,
        "azimuth_radar": AZIMUTH_RADAR,
        "gmti": GMTI,
    },
    default_key="radar",
)
