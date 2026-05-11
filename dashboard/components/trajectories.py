"""
Trajectory ComponentSpecs.

Each `ComponentSpec` registered here describes one of the framework's
trajectory classes in terms of its tunable parameters. For trajectories
whose `__init__` takes nontrivial arguments (state vector + layout
rather than scalars), a custom `build` function assembles them from the
form values.
"""
from __future__ import annotations

import numpy as np

from sdf.core.state import StateLayout
from sdf.scenarios import (
    ConstantVelocityTrajectory,
    MountainPassTrajectory,
)

from dashboard.schema import ComponentChoice, ComponentSpec, ParameterSpec


def _build_constant_velocity(values: dict) -> ConstantVelocityTrajectory:
    """Assemble a CV trajectory from a 3D position + 3D velocity."""
    pos = values["initial_position"]
    vel = values["initial_velocity"]
    # State layout matches what ConstantVelocity motion model uses:
    # [x, vx, y, vy, z, vz] in 3D.
    initial_state = np.array(
        [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]], dtype=float
    )
    layout = StateLayout(dim=3, position_idx=(0, 2, 4), velocity_idx=(1, 3, 5))
    return ConstantVelocityTrajectory(initial_state=initial_state, layout=layout)


MOUNTAIN_PASS = ComponentSpec(
    label="Mountain pass (3D winding road)",
    constructor=MountainPassTrajectory,
    description=(
        "A 3D sinusoidal road of given length, with controllable lateral "
        "and vertical amplitudes. Vehicle moves at constant forward speed."
    ),
    parameters=[
        ParameterSpec(
            "v_kmh", float, default=20.0, min=1.0, max=200.0, step=1.0,
            description="Forward speed along x", unit="km/h",
        ),
        ParameterSpec(
            "length", float, default=10_000.0, min=500.0, max=50_000.0, step=100.0,
            description="Road length along x", unit="m",
        ),
        ParameterSpec(
            "y_amp", float, default=1_000.0, min=0.0, max=5_000.0, step=10.0,
            description="Lateral (y) sinusoid amplitude", unit="m",
        ),
        ParameterSpec(
            "z_amp", float, default=1_000.0, min=0.0, max=5_000.0, step=10.0,
            description="Vertical (z) sinusoid amplitude", unit="m",
        ),
    ],
)


CONSTANT_VELOCITY = ComponentSpec(
    label="Constant velocity (straight line)",
    constructor=ConstantVelocityTrajectory,
    description="Target moves with fixed velocity from a given initial state.",
    build=_build_constant_velocity,
    parameters=[
        ParameterSpec(
            "initial_position", float, default=[0.0, 0.0, 0.0], length=3,
            description="Initial position (x, y, z)", unit="m",
        ),
        ParameterSpec(
            "initial_velocity", float, default=[20.0, 0.0, 0.0], length=3,
            description="Velocity (vx, vy, vz)", unit="m/s",
        ),
    ],
)


TRAJECTORY_CHOICE = ComponentChoice(
    label="Trajectory",
    options={
        "mountain_pass": MOUNTAIN_PASS,
        "constant_velocity": CONSTANT_VELOCITY,
    },
    default_key="mountain_pass",
)
