"""
Filter ComponentSpecs.

Filters differ from other components in that they depend on other
configured pieces — a motion model (configured in the motion-model
section), an initial state (derived from the trajectory's t=0 state
plus user-supplied prior offsets and uncertainties), and optionally a
road map. The simulation runner assembles those dependencies; the
parameters declared here are the filter's own user-controllable knobs.

IMM uses fixed sub-models (CV + CT-left + CT-right, all sharing the 2D
[x, vx, y, vy] layout) with user-editable process noises and transition
probability matrix. Editing the sub-model list itself is deferred.

The schema's `build` field is unused here because filter construction
is too cross-cutting to express in a single function — the runner
handles it.
"""
from __future__ import annotations

from sdf.filters import (
    ExtendedKalmanFilter,
    IMMFilter,
    KalmanFilter,
)

from dashboard.schema import ComponentChoice, ComponentSpec, ParameterSpec


# Prior parameters that every filter shares. Defined once and spliced
# into each filter spec so the form layout is consistent across filter
# types (and the runner can pull them by name).
_PRIOR_PARAMETERS = [
    ParameterSpec(
        "prior_position_offset", float, default=50.0,
        min=0.0, max=1_000.0, step=1.0,
        description="Initial position offset from truth (per axis, m)",
        unit="m",
    ),
    ParameterSpec(
        "prior_position_sigma", float, default=20.0,
        min=0.1, max=500.0, step=0.1,
        description="Initial position uncertainty (1-sigma, per axis)",
        unit="m",
    ),
    ParameterSpec(
        "prior_velocity_sigma", float, default=4.0,
        min=0.01, max=50.0, step=0.01,
        description="Initial velocity uncertainty (1-sigma, per axis)",
        unit="m/s",
    ),
]


KALMAN = ComponentSpec(
    label="Kalman filter (linear)",
    constructor=KalmanFilter,
    description=(
        "Standard linear Kalman filter. Use with linear sensors only "
        "(e.g. Cartesian position); pair with nonlinear sensors will "
        "fail at runtime."
    ),
    parameters=list(_PRIOR_PARAMETERS),
)


EXTENDED_KALMAN = ComponentSpec(
    label="Extended Kalman filter (EKF)",
    constructor=ExtendedKalmanFilter,
    description=(
        "EKF with first-order linearisation of nonlinear measurement "
        "functions. Default choice for radar / GMTI sensors."
    ),
    parameters=list(_PRIOR_PARAMETERS),
)


# Note: there is no separate "road-aided EKF" filter. Road-aiding is a
# property of the run, not the filter: tick 'Road-aided approximation' in the
# Road map section to fuse the road into whichever base filter you pick (KF,
# EKF, or IMM). That keeps "which filter" and "is it road-aided" as two
# independent choices instead of baking them into one dropdown entry.


# IMM sub-models: CV + CT + CA, all on one unified 7-D state
# [x, vx, ax, y, vy, ay, omega] so the mixing step is well-defined. The turn
# mode estimates its own turn rate omega (a single adaptive model handles left
# and right turns), so there is no fixed turn-rate parameter. We expose the
# per-mode process noises and the TPM diagonal (off-diagonals split evenly).
# This pairs naturally with the 'Fighter jet' trajectory, which visits all
# three regimes (and works in 2D or 3D). Stiffer CV noise makes the CV mode
# 'fail' more visibly on maneuvers, sharpening the mode switching.
IMM = ComponentSpec(
    label="IMM (CV + CT + CA, 3 modes)",
    constructor=IMMFilter,
    description=(
        "Interacting Multiple Models filter with three modes on a shared "
        "state: constant velocity (smooth), coordinated turn (adaptive turn "
        "rate), and constant acceleration (fast maneuvers). The mode "
        "probabilities show which dynamics the target is currently in. Pair "
        "with the 'Fighter jet' trajectory to see all three light up."
    ),
    parameters=[
        *_PRIOR_PARAMETERS,
        ParameterSpec(
            "cv_process_noise_std", float, default=0.5,
            min=0.01, max=20.0, step=0.1,
            description="CV mode acceleration std-dev (small = stiff)",
            unit="m/s²",
        ),
        ParameterSpec(
            "ct_process_noise_std", float, default=1.0,
            min=0.01, max=20.0, step=0.1,
            description="CT mode acceleration std-dev", unit="m/s²",
        ),
        ParameterSpec(
            "ct_omega_noise_std", float, default=0.1,
            min=0.001, max=1.0, step=0.005,
            description="CT mode turn-rate random-walk std-dev", unit="rad/s",
        ),
        ParameterSpec(
            "ca_jerk_std", float, default=3.0,
            min=0.01, max=20.0, step=0.1,
            description="CA mode jerk std-dev", unit="m/s³",
        ),
        ParameterSpec(
            "tpm_self_prob", float, default=0.94,
            min=0.5, max=0.999, step=0.001,
            description=(
                "Probability of staying in the same mode (TPM diagonal); "
                "off-diagonals = (1-p) / (n_modes-1)"
            ),
        ),
    ],
)


FILTER_CHOICE = ComponentChoice(
    label="Filter",
    options={
        "kf": KALMAN,
        "ekf": EXTENDED_KALMAN,
        "imm": IMM,
    },
    default_key="ekf",
)
