"""
Platform ComponentSpecs.

Platforms describe how a sensor moves through space. StationaryPlatform
is the default (fixed position); the three from awacs.py give the
AWACS-style flight patterns. All take vector parameters, so each has a
custom build function.
"""
from __future__ import annotations

import numpy as np

from sdf.scenarios.awacs import CircleFlight, RacetrackFlight, StraightFlight
from sdf.scenarios.platform import StationaryPlatform

from dashboard.schema import ComponentChoice, ComponentSpec, ParameterSpec


def _build_stationary(values: dict) -> StationaryPlatform:
    return StationaryPlatform(position=np.asarray(values["position"], dtype=float))


def _build_straight(values: dict) -> StraightFlight:
    return StraightFlight(
        start_position=np.asarray(values["start_position"], dtype=float),
        velocity=np.asarray(values["velocity"], dtype=float),
    )


def _build_circle(values: dict) -> CircleFlight:
    return CircleFlight(
        center=np.asarray(values["center"], dtype=float),
        radius=values["radius"],
        speed=values["speed"],
        phase=values["phase"],
    )


def _build_racetrack(values: dict) -> RacetrackFlight:
    return RacetrackFlight(
        center=np.asarray(values["center"], dtype=float),
        leg_length=values["leg_length"],
        radius=values["radius"],
        speed=values["speed"],
        phase_arc_length=values["phase_arc_length"],
    )


STATIONARY = ComponentSpec(
    label="Stationary",
    constructor=StationaryPlatform,
    build=_build_stationary,
    parameters=[
        ParameterSpec("position", float, default=[0.0, 10_000.0, 100.0], length=3,
                      description="Fixed sensor position (x, y, z)", unit="m"),
    ],
)


STRAIGHT_FLIGHT = ComponentSpec(
    label="Straight flight",
    constructor=StraightFlight,
    build=_build_straight,
    parameters=[
        ParameterSpec("start_position", float,
                      default=[0.0, 0.0, 10_000.0], length=3,
                      description="Start position (x, y, z)", unit="m"),
        ParameterSpec("velocity", float,
                      default=[200.0, 0.0, 0.0], length=3,
                      description="Velocity (vx, vy, vz)", unit="m/s"),
    ],
)


CIRCLE_FLIGHT = ComponentSpec(
    label="Circle flight",
    constructor=CircleFlight,
    build=_build_circle,
    parameters=[
        ParameterSpec("center", float, default=[0.0, 0.0, 10_000.0], length=3,
                      description="Orbit centre (x, y, z)", unit="m"),
        ParameterSpec("radius", float, default=20_000.0,
                      min=1_000.0, max=200_000.0, step=100.0,
                      description="Orbit radius", unit="m"),
        ParameterSpec("speed", float, default=200.0,
                      min=10.0, max=500.0, step=1.0,
                      description="Tangential speed", unit="m/s"),
        ParameterSpec("phase", float, default=0.0,
                      min=0.0, max=6.2832, step=0.01,
                      description="Initial phase angle", unit="rad"),
    ],
)


RACETRACK_FLIGHT = ComponentSpec(
    label="Racetrack flight",
    constructor=RacetrackFlight,
    build=_build_racetrack,
    parameters=[
        ParameterSpec("center", float, default=[0.0, 0.0, 10_000.0], length=3,
                      description="Pattern centre (x, y, z)", unit="m"),
        ParameterSpec("leg_length", float, default=30_000.0,
                      min=1_000.0, max=200_000.0, step=100.0,
                      description="Straight leg length", unit="m"),
        ParameterSpec("radius", float, default=10_000.0,
                      min=500.0, max=100_000.0, step=100.0,
                      description="Turn radius at each end", unit="m"),
        ParameterSpec("speed", float, default=200.0,
                      min=10.0, max=500.0, step=1.0,
                      description="Speed (constant)", unit="m/s"),
        ParameterSpec("phase_arc_length", float, default=0.0,
                      min=0.0, max=200_000.0, step=100.0,
                      description="Initial arc-length along the pattern", unit="m"),
    ],
)


PLATFORM_CHOICE = ComponentChoice(
    label="Platform",
    options={
        "stationary": STATIONARY,
        "straight": STRAIGHT_FLIGHT,
        "circle": CIRCLE_FLIGHT,
        "racetrack": RACETRACK_FLIGHT,
    },
    default_key="stationary",
)
