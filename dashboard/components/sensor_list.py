"""
SensorListSpec — a list of configured sensors.

The dashboard's only variable-length config: zero or more sensors, each
of which is a (type, parameters) pair. Two reasons it can't be a regular
ComponentSpec:
  1. Length isn't fixed at schema time.
  2. The "kind" of each entry varies — entry 0 might be a radar, entry 1
     a GMTI, entry 2 a Cartesian sensor.

The serialized form (what the dashboard's Store holds) is a list of
dicts:

    [
        {"type": "radar", "params": {"sensor_id": "radar_a", ...}},
        {"type": "gmti",  "params": {"sensor_id": "gmti_b",  ...}},
    ]

`SensorListSpec` provides round-trip helpers (defaults, validate, build)
that operate on this list shape. The UI side knows to render add/remove
controls; the runner iterates the resulting sensor objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dashboard.components.sensors import SENSOR_CHOICE
from dashboard.schema import ComponentChoice


@dataclass
class SensorListSpec:
    """
    Schema for the variable-length list of sensors.

    `choice` is the `ComponentChoice` describing the per-entry sensor
    types; here it's always `SENSOR_CHOICE`, but we keep it as a field
    for testability.
    """

    choice: ComponentChoice = field(default_factory=lambda: SENSOR_CHOICE)
    default_entries: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "type": "radar",
                "params": {
                    "sensor_id": "radar_a",
                    "position": [0.0, 10_000.0, 100.0],
                    "range_std": 80.0,
                    "bearing_std": 8e-3,
                    "elevation_std": 8e-3,
                    "detection_prob": 1.0,
                },
            },
            {
                "type": "radar",
                "params": {
                    "sensor_id": "radar_b",
                    "position": [10_000.0, 0.0, 100.0],
                    "range_std": 80.0,
                    "bearing_std": 8e-3,
                    "elevation_std": 8e-3,
                    "detection_prob": 1.0,
                },
            },
        ]
    )

    def defaults(self) -> list[dict[str, Any]]:
        """A fresh default list (deep-ish copy)."""
        return [
            {"type": e["type"], "params": dict(e["params"])}
            for e in self.default_entries
        ]

    def validate(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Validate each entry's params against its sensor-type spec.

        Returns a list of `{type, params}` dicts with coerced parameter
        values. Unknown types raise KeyError.
        """
        out = []
        for i, entry in enumerate(raw):
            if "type" not in entry or "params" not in entry:
                raise ValueError(
                    f"sensor entry {i}: must have 'type' and 'params'; "
                    f"got {entry!r}"
                )
            spec = self.choice.get(entry["type"])
            out.append({
                "type": entry["type"],
                "params": spec.validate(entry["params"]),
            })
        return out

    def build(self, raw: list[dict[str, Any]]) -> list:
        """Validate and construct all sensors. Returns a list of sensor objects."""
        validated = self.validate(raw)
        return [
            self.choice.get(entry["type"]).construct(entry["params"])
            for entry in validated
        ]


SENSOR_LIST = SensorListSpec()
