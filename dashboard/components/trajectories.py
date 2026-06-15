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
    FighterJetTrajectory,
    MountainPassTrajectory,
)

from dashboard.schema import ComponentChoice, ComponentSpec, ParameterSpec


def _build_constant_velocity(values: dict) -> ConstantVelocityTrajectory:
    """
    Assemble a CV trajectory from a position + velocity.

    Honors a `dim` choice (2D or 3D). A 2D trajectory is the scenario you
    need to exercise the inherently-2D filters (coordinated turn, IMM); a
    3D trajectory pairs with the 3D radar/GMTI sensors. The interleaved
    state layout matches what the ConstantVelocity motion model uses:
    2D -> [x, vx, y, vy], 3D -> [x, vx, y, vy, z, vz].
    """
    dim = int(values.get("dim", 3))
    pos = values["initial_position"]
    vel = values["initial_velocity"]
    if dim == 2:
        initial_state = np.array([pos[0], vel[0], pos[1], vel[1]], dtype=float)
        layout = StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))
    else:
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
            "dim", int, default=3,
            choices=[("2D", 2), ("3D", 3)],
            description="State-space dimension (use 2D for IMM / CT filters)",
        ),
        ParameterSpec(
            "initial_position", float, default=[0.0, 0.0, 0.0], length=3,
            description="Initial position (x, y, z; z ignored in 2D)", unit="m",
        ),
        ParameterSpec(
            "initial_velocity", float, default=[20.0, 0.0, 0.0], length=3,
            description="Velocity (vx, vy, vz)", unit="m/s",
        ),
    ],
)


FIGHTER_JET = ComponentSpec(
    label="Fighter jet (3D hard maneuvers)",
    constructor=FighterJetTrajectory,
    description=(
        "An aggressive 3D target: level flight, a hard left break turn, an "
        "accelerating zoom climb, a harder right break turn, a diving "
        "deceleration, then level again. The hard turns and abrupt "
        "climb/dive break any single motion model — a plain CV filter drifts "
        "hundreds of metres while the CV+CT+CA IMM stays locked on and its "
        "mode probabilities track the maneuver. Pair it with the IMM filter; "
        "in 3D it works with the default radars."
    ),
    parameters=[
        ParameterSpec("dim", int, default=3,
                      choices=[("2D", 2), ("3D", 3)],
                      description="State-space dimension (match your sensors)"),
        ParameterSpec("speed", float, default=200.0, min=20.0, max=600.0, step=5.0,
                      description="Cruise speed", unit="m/s"),
        ParameterSpec("turn_rate", float, default=0.18, min=0.02, max=0.5, step=0.01,
                      description="Break-turn rate (harder = sharper)", unit="rad/s"),
        ParameterSpec("tangential_accel", float, default=22.0,
                      min=1.0, max=100.0, step=1.0,
                      description="Accel/decel magnitude along heading", unit="m/s²"),
        ParameterSpec("vertical_accel", float, default=18.0,
                      min=0.0, max=100.0, step=1.0,
                      description="Climb/dive acceleration (3D only)", unit="m/s²"),
    ],
)


TRAJECTORY_CHOICE = ComponentChoice(
    label="Trajectory",
    options={
        "mountain_pass": MOUNTAIN_PASS,
        "constant_velocity": CONSTANT_VELOCITY,
        "fighter_jet": FIGHTER_JET,
    },
    default_key="mountain_pass",
)
