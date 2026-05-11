"""
Filter ComponentSpecs.

Filters differ from other components in that they depend on other
configured pieces — a motion model (configured in the motion-model
section), an initial state (derived from the trajectory's t=0 state
plus user-supplied prior offsets and uncertainties), and optionally a
road map. The simulation runner assembles those dependencies; the
parameters declared here are the filter's own user-controllable knobs.

Per the v6 agreement, IMM uses fixed sub-models (CV + CA + CT-known)
with user-editable process noises and transition probability matrix.
Editing the sub-model list itself is deferred.

The schema's `build` field is unused here because filter construction
is too cross-cutting to express in a single function — the runner
handles it.
"""
from __future__ import annotations

from sdf.filters import (
    ExtendedKalmanFilter,
    IMMFilter,
    KalmanFilter,
    RoadAidedExtendedKalmanFilter,
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


ROAD_AIDED_EKF = ComponentSpec(
    label="Road-aided EKF",
    constructor=RoadAidedExtendedKalmanFilter,
    description=(
        "EKF augmented with a fictitious cross-track measurement from "
        "the polygonal road map, applied at every step. Requires the "
        "road map to be enabled."
    ),
    parameters=list(_PRIOR_PARAMETERS),
)


# IMM has its own extra parameters on top of the shared prior fields.
# The sub-model list is fixed: CV + CA + CT-known.
# We expose:
#   - process noise for each sub-model (3 floats)
#   - omega for the CT sub-model (1 float)
#   - TPM diagonal — probability of staying in the same mode (one float
#     per row; off-diagonals are split evenly among other modes). This
#     is a simplification over a fully editable 3x3 matrix and is enough
#     to demonstrate the IMM behaviour while keeping the UI tractable.
IMM = ComponentSpec(
    label="IMM (CV + CA + CT, fixed sub-models)",
    constructor=IMMFilter,
    description=(
        "Interacting Multiple Models filter with three fixed sub-models. "
        "Process noises and the diagonal of the transition probability "
        "matrix are user-controllable; off-diagonals are split evenly. "
        "Sub-model selection itself is fixed in v6."
    ),
    parameters=[
        *_PRIOR_PARAMETERS,
        ParameterSpec(
            "cv_process_noise_std", float, default=2.0,
            min=0.01, max=20.0, step=0.1,
            description="CV sub-model acceleration std-dev", unit="m/s²",
        ),
        ParameterSpec(
            "ca_jerk_std", float, default=2.0,
            min=0.01, max=20.0, step=0.1,
            description="CA sub-model jerk std-dev", unit="m/s³",
        ),
        ParameterSpec(
            "ct_omega", float, default=0.05,
            min=-0.5, max=0.5, step=0.005,
            description="CT sub-model turn rate", unit="rad/s",
        ),
        ParameterSpec(
            "ct_process_noise_std", float, default=1.0,
            min=0.01, max=20.0, step=0.1,
            description="CT sub-model acceleration std-dev", unit="m/s²",
        ),
        ParameterSpec(
            "tpm_self_prob", float, default=0.95,
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
        "road_aided_ekf": ROAD_AIDED_EKF,
        "imm": IMM,
    },
    default_key="ekf",
)
