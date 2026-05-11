"""
Tests for the dashboard schema layer.

This is pure data with no Dash dependency, so it can sit in the main
test suite without dragging in a heavyweight UI library.

Verify:
  1. ParameterSpec.validate coerces scalar inputs to the declared kind.
  2. ParameterSpec.validate enforces vector length.
  3. ComponentSpec.construct calls the constructor with validated kwargs.
  4. ComponentSpec.defaults returns a dict of parameter defaults.
  5. ComponentSpec.validate fills in missing fields from defaults.
  6. ComponentSpec.build overrides the default constructor call when set.
  7. ComponentChoice.get raises on unknown keys.
"""
import pytest

# The dashboard module sits outside src/sdf/, so we add it to the path
# explicitly here. Conftest in this dir handles that.
from dashboard.schema import (
    ComponentChoice,
    ComponentSpec,
    ParameterSpec,
)


class TestParameterSpec:
    def test_scalar_coerces_to_kind(self):
        p = ParameterSpec("v", float, default=0.0)
        assert p.validate("3.5") == 3.5
        assert isinstance(p.validate("3.5"), float)

    def test_int_kind_coerces(self):
        p = ParameterSpec("n", int, default=0)
        assert p.validate("7") == 7
        assert isinstance(p.validate("7"), int)

    def test_bool_kind(self):
        p = ParameterSpec("on", bool, default=False)
        # Python's bool() coerces non-empty truthy values.
        assert p.validate(True) is True
        assert p.validate(0) is False

    def test_vector_validates_length(self):
        p = ParameterSpec("pos", float, default=[0.0, 0.0, 0.0], length=3)
        assert p.validate([1, 2, 3]) == [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="length-3"):
            p.validate([1, 2])
        with pytest.raises(ValueError, match="length-3"):
            p.validate(5.0)


class TestComponentSpec:
    def test_construct_calls_constructor(self):
        class Widget:
            def __init__(self, a, b):
                self.a = a
                self.b = b

        spec = ComponentSpec(
            label="Widget",
            constructor=Widget,
            parameters=[
                ParameterSpec("a", float, default=1.0),
                ParameterSpec("b", int, default=2),
            ],
        )
        w = spec.construct({"a": 3.0, "b": 4})
        assert w.a == 3.0 and w.b == 4

    def test_defaults_returns_defaults(self):
        spec = ComponentSpec(
            label="X",
            constructor=lambda **kw: kw,
            parameters=[
                ParameterSpec("p", float, default=1.5),
                ParameterSpec("q", int, default=7),
                ParameterSpec("v", float, default=[0.0, 1.0, 2.0], length=3),
            ],
        )
        d = spec.defaults()
        assert d == {"p": 1.5, "q": 7, "v": [0.0, 1.0, 2.0]}

    def test_validate_fills_missing(self):
        spec = ComponentSpec(
            label="X",
            constructor=lambda **kw: kw,
            parameters=[
                ParameterSpec("a", float, default=1.0),
                ParameterSpec("b", int, default=2),
            ],
        )
        # Only `a` provided; `b` should fall through to its default.
        v = spec.validate({"a": "5.5"})
        assert v == {"a": 5.5, "b": 2}

    def test_build_overrides_constructor(self):
        spec = ComponentSpec(
            label="X",
            constructor=lambda **kw: kw,
            parameters=[ParameterSpec("a", float, default=1.0)],
            build=lambda values: {"doubled": values["a"] * 2},
        )
        assert spec.construct({"a": 3.0}) == {"doubled": 6.0}


class TestComponentChoice:
    def test_get_returns_spec(self):
        spec_a = ComponentSpec(label="A", constructor=lambda: "a")
        spec_b = ComponentSpec(label="B", constructor=lambda: "b")
        choice = ComponentChoice(
            label="trajectory",
            options={"a": spec_a, "b": spec_b},
            default_key="a",
        )
        assert choice.get("a") is spec_a
        assert choice.get("b") is spec_b

    def test_unknown_key_raises(self):
        choice = ComponentChoice(
            label="trajectory",
            options={"a": ComponentSpec(label="A", constructor=lambda: 1)},
            default_key="a",
        )
        with pytest.raises(KeyError, match="unknown"):
            choice.get("z")

    def test_keys_returns_option_names(self):
        choice = ComponentChoice(
            label="trajectory",
            options={
                "a": ComponentSpec(label="A", constructor=lambda: 1),
                "b": ComponentSpec(label="B", constructor=lambda: 2),
            },
            default_key="a",
        )
        assert set(choice.keys()) == {"a", "b"}
