sensor-data-fusion/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── scenarios/
│   │   ├── aircraft_imm_3d.yaml
│   │   ├── ground_target_gmti.yaml
│   │   └── occlusion_demo_2d.yaml
│   └── sensors/
│       ├── radar_2d.yaml
│       ├── gmti.yaml
│       └── passive_bearing.yaml
├── src/
│   └── sdf/                          # the package
│       ├── core/
│       │   ├── state.py              # State, StateDistribution (mean+cov)
│       │   ├── measurement.py        # Measurement dataclass
│       │   ├── track.py              # Track (history of states)
│       │   └── time.py               # TimeStamp utilities
│       ├── motion_models/
│       │   ├── base.py               # MotionModel ABC
│       │   ├── constant_velocity.py  # CV (2D/3D)
│       │   ├── constant_accel.py     # CA
│       │   ├── constant_turn.py      # CT (2D), coordinated turn (3D)
│       │   └── singer.py             # Singer accel model (optional)
│       ├── sensors/
│       │   ├── base.py               # Sensor ABC
│       │   ├── radar.py              # range/bearing/(elevation)
│       │   ├── gmti.py               # range/bearing/range-rate + MDV
│       │   ├── cartesian.py          # direct position (toy)
│       │   └── occlusion.py          # OcclusionModel ABC + terrain/MDV
│       ├── filters/
│       │   ├── base.py               # Filter ABC (predict/update interface)
│       │   ├── kalman.py             # KF
│       │   ├── extended_kalman.py    # EKF
│       │   ├── unscented_kalman.py   # UKF
│       │   ├── imm.py                # IMM (composes filters)
│       │   ├── ekf_road.py           # road-constrained EKF
│       │   └── smoothers.py          # RTS, IMM smoother
│       ├── data_association/
│       │   ├── base.py               # Associator ABC
│       │   ├── nn.py                 # Nearest-neighbor / GNN
│       │   ├── jpda.py               # JPDA
│       │   └── mht.py                # (planned)
│       ├── scenarios/
│       │   ├── base.py               # Scenario ABC, Trajectory generators
│       │   ├── trajectory.py         # piecewise CV/CA/CT trajectories
│       │   ├── terrain.py            # terrain / road network loader
│       │   └── builder.py            # YAML → Scenario object
│       ├── metrics/
│       │   ├── rmse.py
│       │   ├── nees.py
│       │   └── ospa.py
│       ├── simulation/
│       │   ├── engine.py             # SimulationEngine: orchestrates run
│       │   └── runner.py             # high-level run_scenario()
│       └── viz/
│           ├── plot_2d.py
│           ├── plot_3d.py
│           └── animate.py
├── app/                              # Streamlit lives here
│   ├── Home.py
│   ├── pages/
│   │   ├── 1_Scenario_Builder.py
│   │   ├── 2_Run_Simulation.py
│   │   ├── 3_Compare_Filters.py
│   │   └── 4_Retrodiction.py
│   └── components/
│       └── plots.py
├── notebooks/
│   ├── 01_kf_basics.ipynb
│   ├── 02_imm_aircraft.ipynb
│   ├── 03_gmti_road.ipynb
│   └── 04_retrodiction.ipynb
├── tests/
│   ├── test_motion_models.py
│   ├── test_sensors.py
│   ├── test_filters.py
│   └── test_imm_smoother.py
└── results/                          # Saved figures, GIFs for README