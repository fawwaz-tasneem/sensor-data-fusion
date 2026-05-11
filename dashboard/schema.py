"""
Schema for configurable framework components.

The dashboard configures SDF framework objects (trajectories, sensors,
filters, motion models, ...) through forms that are generated from
schemas rather than hand-coded per class. A `ComponentSpec` describes
one configurable class — its label, its constructor, and its parameters.
A `ParameterSpec` describes one tunable scalar (or vector) parameter.

The UI layer reads these specs to build Dash forms; the simulation
runner reads them to build framework objects from the form values.

This separation means:
  * Adding a new trajectory / sensor / filter type to the dashboard is
    a one-place change: add a ComponentSpec to the relevant registry.
  * The UI generator doesn't need to know which classes exist.
  * Validation lives in one place (here), not scattered through
    callbacks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union


# A parameter is either a scalar (float/int/bool) or a small fixed-length
# vector (e.g. a 3D position). We keep the distinction simple: `length=1`
# means scalar, `length>1` means a vector input with that many slots.
ParameterKind = Union[type[float], type[int], type[bool], type[str]]


@dataclass
class ParameterSpec:
    """
    One tunable parameter of a component.

    For scalar parameters, `length=1` and the UI renders one slider/input.
    For vector parameters (e.g. a 3D position), `length=3` renders three
    side-by-side inputs sharing a label.

    `choices` overrides the slider/input UI with a dropdown — used for
    enum-like parameters where only a few values make sense.
    """

    name: str
    kind: ParameterKind = float
    default: Any = 0.0
    description: str = ""

    # Numeric bounds (used by sliders, ignored for bool/str).
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None

    # Vector parameters: length > 1 means a fixed-length vector.
    length: int = 1

    # If set, parameter is rendered as a dropdown rather than a slider.
    choices: Optional[list[tuple[str, Any]]] = None

    # Optional unit string shown after the input (e.g. "m", "km/h", "rad").
    unit: str = ""

    def validate(self, value: Any) -> Any:
        """Validate and coerce a raw form value to the parameter's kind."""
        if self.length > 1:
            if not isinstance(value, (list, tuple)) or len(value) != self.length:
                raise ValueError(
                    f"{self.name}: expected length-{self.length} vector, "
                    f"got {value!r}"
                )
            return [self.kind(v) for v in value]
        return self.kind(value)


@dataclass
class ComponentSpec:
    """
    Description of one configurable class.

    `constructor` is called with the keyword arguments derived from the
    parameter form to produce a framework object. `build` lets a spec
    override the default `constructor(**params)` call when a class needs
    custom assembly (e.g. derived parameters, kwargs that aren't direct
    form fields).
    """

    label: str
    constructor: Callable[..., Any]
    parameters: list[ParameterSpec] = field(default_factory=list)
    build: Optional[Callable[[dict[str, Any]], Any]] = None
    description: str = ""

    def construct(self, values: dict[str, Any]) -> Any:
        """Build a framework object from a dict of validated form values."""
        if self.build is not None:
            return self.build(values)
        return self.constructor(**values)

    def defaults(self) -> dict[str, Any]:
        """Default values keyed by parameter name."""
        out = {}
        for p in self.parameters:
            if p.length > 1:
                out[p.name] = list(p.default)
            else:
                out[p.name] = p.default
        return out

    def validate(self, raw_values: dict[str, Any]) -> dict[str, Any]:
        """Validate a dict of raw form values, returning coerced values."""
        out = {}
        for p in self.parameters:
            if p.name not in raw_values:
                if p.length > 1:
                    out[p.name] = list(p.default)
                else:
                    out[p.name] = p.default
            else:
                out[p.name] = p.validate(raw_values[p.name])
        return out


@dataclass
class ComponentChoice:
    """
    A registry of one-of-many component specs (e.g. all trajectory types).

    The UI renders a dropdown choosing between specs, then a parameter
    form for the selected spec. `key` is the value stored in the
    dropdown (e.g. "mountain_pass"); `default_key` is the spec selected
    on load.
    """

    label: str
    options: dict[str, ComponentSpec]
    default_key: str

    def get(self, key: str) -> ComponentSpec:
        if key not in self.options:
            raise KeyError(
                f"unknown {self.label} key {key!r}; "
                f"options: {list(self.options)}"
            )
        return self.options[key]

    def keys(self) -> list[str]:
        return list(self.options.keys())
