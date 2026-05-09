# Sensor Data Fusion Tracking Framework

Modular Python framework for sensor data fusion and target tracking. Built to
extend with additional filters (EKF, UKF, IMM), sensors (radar, GMTI), and
data association methods (GNN, JPDA, MHT).

## Quick start

```bash
# Install in editable mode
pip install -e .

# Run the minimal example
python examples/minimal_kf_2d.py

# Run the test suite
pytest -v
```

## Layout

```
src/sdf/
├── core/           # State, Measurement, Track
├── motion_models/  # CV (CA, CT, ... to come)
├── sensors/        # Cartesian (radar, GMTI to come)
├── filters/        # KF (EKF, UKF, IMM to come)
├── scenarios/      # Trajectory generators
└── simulation/     # SimulationEngine
```

## Architecture

The framework rests on a few small interfaces:

- `MotionModel` — defines `f`, `F`, `Q`, used by all filters.
- `Sensor` — defines `h`, `H`, plus a `measure()` pipeline with detection
  probability and pluggable occlusion.
- `Filter` — `predict`, `update`, `step`. Filters consume a `MotionModel`
  and a `Sensor`, never know about ground truth.
- `SimulationEngine` — wires trajectory + sensors + filter and produces a
  `SimulationResult` for plotting and metrics.

Adding a new filter or sensor means writing one new class implementing the
relevant ABC.
