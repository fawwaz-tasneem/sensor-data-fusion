"""
Occlusion ComponentSpecs.

Both built-in occlusion types have cross-component dependencies:
  * TunnelOcclusion needs the road map (which itself is built from the
    trajectory in the canonical examples).
  * DopplerBlindnessOcclusion is conceptually per-sensor (it uses the
    sensor's own position and velocity to compute the clutter notch).

The specs here describe each occlusion's *own* parameters (arc-length
range, radius; MDV, floor). The simulation runner wires in the road or
sensor position when assembling the scenario.

Tunnel arc-length range is parameterized as *fractions* of the road's
total length rather than absolute metres, so the same tunnel
configuration is meaningful across different trajectory lengths.
"""
from __future__ import annotations

from sdf.sensors import DopplerBlindnessOcclusion, TunnelOcclusion

from dashboard.schema import ComponentChoice, ComponentSpec, ParameterSpec


# Tunnel arc-length range is given as fractions (0..1) of the road's
# total length. The runner converts these to absolute l_in / l_out using
# the road map. There's no `build` here because the runner constructs
# the TunnelOcclusion directly with the resolved values.
TUNNEL = ComponentSpec(
    label="Tunnel (road-aligned occlusion)",
    constructor=TunnelOcclusion,
    description=(
        "Tubular occlusion along a road segment. Targets inside the tube "
        "are hidden from any sensor that carries this occlusion model."
    ),
    parameters=[
        ParameterSpec("l_in_frac", float, default=0.40,
                      min=0.0, max=1.0, step=0.01,
                      description="Tunnel entry, as fraction of road length"),
        ParameterSpec("l_out_frac", float, default=0.60,
                      min=0.0, max=1.0, step=0.01,
                      description="Tunnel exit, as fraction of road length"),
        ParameterSpec("radius", float, default=30.0,
                      min=1.0, max=500.0, step=1.0,
                      description="Cross-track radius of the tunnel tube", unit="m"),
    ],
)


# Doppler blindness uses sensor position/velocity, which the runner
# injects. We expose MDV and pd_floor.
DOPPLER_BLINDNESS = ComponentSpec(
    label="Doppler clutter notch",
    constructor=DopplerBlindnessOcclusion,
    description=(
        "Probabilistic occlusion: targets with low radial velocity relative "
        "to the sensor are buried in clutter and missed."
    ),
    parameters=[
        ParameterSpec("mdv", float, default=3.0,
                      min=0.1, max=50.0, step=0.1,
                      description="Minimum detectable radial velocity",
                      unit="m/s"),
        ParameterSpec("pd_floor", float, default=0.05,
                      min=0.0, max=1.0, step=0.01,
                      description="Floor on detection probability inside the notch"),
    ],
)


OCCLUSION_CHOICE = ComponentChoice(
    label="Occlusion",
    options={
        "none": ComponentSpec(
            label="None",
            constructor=lambda: None,
            description="No occlusion — sensors always see the target.",
        ),
        "tunnel": TUNNEL,
        "doppler": DOPPLER_BLINDNESS,
    },
    default_key="none",
)
