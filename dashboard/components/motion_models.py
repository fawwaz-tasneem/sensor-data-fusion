"""
Motion model ComponentSpecs.

The framework's four motion models: ConstantVelocity, ConstantAcceleration,
CoordinatedTurn (known turn rate), and CoordinatedTurnUnknown (turn rate
estimated). CT is 2D-only; the others support 2D and 3D.
"""
from __future__ import annotations

from sdf.motion_models import (
    ConstantAcceleration,
    ConstantVelocity,
    CoordinatedTurn,
    CoordinatedTurnUnknown,
)

from dashboard.schema import ComponentChoice, ComponentSpec, ParameterSpec


CONSTANT_VELOCITY = ComponentSpec(
    label="Constant velocity (CV)",
    constructor=ConstantVelocity,
    description="Nearly-constant-velocity model with white-noise acceleration.",
    parameters=[
        ParameterSpec(
            "dim", int, default=3,
            choices=[("2D", 2), ("3D", 3)],
            description="State-space dimension",
        ),
        ParameterSpec(
            "process_noise_std", float, default=2.0,
            min=0.01, max=20.0, step=0.1,
            description="White acceleration std-dev", unit="m/s²",
        ),
    ],
)


CONSTANT_ACCELERATION = ComponentSpec(
    label="Constant acceleration (CA)",
    constructor=ConstantAcceleration,
    description="Nearly-constant-acceleration model with white-noise jerk.",
    parameters=[
        ParameterSpec(
            "dim", int, default=3,
            choices=[("2D", 2), ("3D", 3)],
            description="State-space dimension",
        ),
        ParameterSpec(
            "jerk_std", float, default=2.0,
            min=0.01, max=20.0, step=0.1,
            description="White jerk std-dev", unit="m/s³",
        ),
    ],
)


COORDINATED_TURN = ComponentSpec(
    label="Coordinated turn (CT, known omega)",
    constructor=CoordinatedTurn,
    description="2D constant-rate turn with fixed angular velocity.",
    parameters=[
        ParameterSpec(
            "omega", float, default=0.05,
            min=-0.5, max=0.5, step=0.005,
            description="Turn rate (positive = CCW)", unit="rad/s",
        ),
        ParameterSpec(
            "process_noise_std", float, default=1.0,
            min=0.01, max=20.0, step=0.1,
            description="White acceleration std-dev", unit="m/s²",
        ),
    ],
)


COORDINATED_TURN_UNKNOWN = ComponentSpec(
    label="Coordinated turn (CT, unknown omega)",
    constructor=CoordinatedTurnUnknown,
    description="2D turn with the turn rate estimated as part of the state.",
    parameters=[
        ParameterSpec(
            "process_noise_std", float, default=1.0,
            min=0.01, max=20.0, step=0.1,
            description="White acceleration std-dev", unit="m/s²",
        ),
        ParameterSpec(
            "omega_noise_std", float, default=0.05,
            min=0.001, max=0.5, step=0.005,
            description="Random walk std-dev on the turn rate", unit="rad/s",
        ),
    ],
)


MOTION_MODEL_CHOICE = ComponentChoice(
    label="Motion model",
    options={
        "cv": CONSTANT_VELOCITY,
        "ca": CONSTANT_ACCELERATION,
        "ct_known": COORDINATED_TURN,
        "ct_unknown": COORDINATED_TURN_UNKNOWN,
    },
    default_key="cv",
)
