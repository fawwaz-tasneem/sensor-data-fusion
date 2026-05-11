"""
Sensor ComponentSpecs.

Each sensor has its own parameter form. RadarSensor and GMTIRadarSensor
take a 3D position as `np.ndarray`, so we use a custom `build` to convert
from form list to ndarray. Occlusion model is wired in by the simulation
runner, not here, because it's a cross-component dependency.
"""
from __future__ import annotations

import numpy as np

from sdf.sensors import CartesianPositionSensor, GMTIRadarSensor, RadarSensor

from dashboard.schema import ComponentChoice, ComponentSpec, ParameterSpec


def _build_radar(values: dict) -> RadarSensor:
    return RadarSensor(
        sensor_id=values["sensor_id"],
        position=np.asarray(values["position"], dtype=float),
        range_std=values["range_std"],
        bearing_std=values["bearing_std"],
        elevation_std=values["elevation_std"],
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
    description="Ground moving target indicator radar; reports range-rate in addition to range/azimuth/elevation.",
    parameters=[
        ParameterSpec("sensor_id", str, default="gmti_a",
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
        ParameterSpec("range_rate_std", float, default=0.5,
                      min=0.01, max=10.0, step=0.01,
                      description="Range-rate noise std-dev", unit="m/s"),
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
        "gmti": GMTI,
    },
    default_key="radar",
)
