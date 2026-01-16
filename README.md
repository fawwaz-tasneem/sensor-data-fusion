# ISDF: Introduction to Sensor Data Fusion

A sensor fusion framework demonstrating state estimation using a Kalman filter to track a vehicle traveling through a mountain pass using dual radar sensors.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Ground Truth Model](#ground-truth-model)
- [Sensor Model](#sensor-model)
- [Kalman Filter Implementation](#kalman-filter-implementation)
- [State Representation](#state-representation)
- [Key Matrices](#key-matrices)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [Usage](#usage)

## Overview

This project implements a real-time sensor fusion pipeline that:

1. Simulates a vehicle traveling on a 10 km mountain pass at 20 km/h with sinusoidal lateral and vertical motion
2. Generates noisy range and azimuth measurements from two fixed radar stations
3. Fuses measurements using a linear Kalman filter to estimate 3D position, velocity, and acceleration
4. Visualizes ground truth, sensor measurements, and filter estimates in an interactive 3D environment

## System Architecture

```
Ground Truth (Environment)
        ↓
Dual Radar Sensors (Polar measurements: r, φ)
        ↓
Polar-to-Cartesian Conversion (nonlinear approximation)
        ↓
Linear Kalman Filter (state fusion and estimation)
        ↓
3D Visualization & Performance Analysis
```

## Ground Truth Model

The vehicle trajectory is defined by the `MountainPassRoad` class, which generates the true position as a function of time:

$$x(t) = v \cdot t$$

$$y(t) = a_y \sin\left(\frac{4\pi v}{a_x} t\right)$$

$$z(t) = a_z \sin\left(\frac{\pi v}{a_x} t\right)$$

**Parameters:**
- Velocity: $v = 20$ km/h = 5.56 m/s
- Road length: $a_x = 10$ km
- Lateral amplitude: $a_y = 1$ km
- Vertical amplitude: $a_z = 1$ km
- Time step: $\Delta t = 2$ s
- Total simulation time: $T_{stop} = 6480$ s (road length / velocity)

This produces a realistic mountain pass trajectory with combined sinusoidal oscillations in both lateral (Y) and vertical (Z) directions.

## Sensor Model

### Radar Configuration

Two radar stations are positioned at fixed locations:
- **Radar 1:** $(x_s, y_s, z_s) = (0, 100000, 10000)$ meters
- **Radar 2:** $(x_s, y_s, z_s) = (100000, 0, 10000)$ meters

### Measurement Equations (Polar Coordinates)

Each radar measures range and azimuth:

$$r = \sqrt{(x_t - x_s)^2 + (y_t - y_s)^2 + (z_t - z_s)^2 - z_s^2}$$

$$\phi = \arctan2(y_t - y_s, x_t - x_s)$$

where $(x_t, y_t, z_t)$ is the target position and $(x_s, y_s, z_s)$ is the radar station position.

### Measurement Noise

Both measurements are corrupted by Gaussian white noise:
- Range noise: $\sigma_r = 10$ m
- Azimuth noise: $\sigma_\phi = 0.1°$

The measurement noise covariance in polar coordinates:

$$\mathbf{R}_{\text{polar}} = \begin{bmatrix} \sigma_r^2 & 0 \\ 0 & \sigma_\phi^2 \end{bmatrix}$$

### Conversion to Cartesian Coordinates

Measurements are converted from polar to Cartesian for linear filtering:

$$x_{\text{meas}} = x_s + r \cos(\phi)$$

$$y_{\text{meas}} = y_s + r \sin(\phi)$$

The measurement noise is transformed to Cartesian using a Jacobian approximation:

$$\text{Var}(x_{\text{meas}}) = (\sigma_r \cos\phi)^2 + (r \sin\phi \cdot \sigma_\phi)^2$$

$$\text{Var}(y_{\text{meas}}) = (\sigma_r \sin\phi)^2 + (r \cos\phi \cdot \sigma_\phi)^2$$

Note: This transformation is an approximation suitable for linear KF. An Extended Kalman Filter (EKF) or Unscented Kalman Filter (UKF) would handle nonlinearity more rigorously.

## Kalman Filter Implementation

The linear Kalman filter performs state estimation in two phases: prediction and update.

### Predict Phase (Time Update)

Propagates the state estimate forward in time using the constant acceleration model:

$$\mathbf{x}_{k|k-1} = \mathbf{F} \mathbf{x}_{k-1|k-1}$$

$$\mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{D}$$

Where:
- $\mathbf{x}_{k|k-1}$ is the predicted state
- $\mathbf{P}_{k|k-1}$ is the predicted state covariance
- $\mathbf{F}$ is the state transition matrix
- $\mathbf{D}$ is the process noise covariance

### Update Phase (Measurement Update)

Corrects the state estimate based on measurement residuals:

$$\boldsymbol{\nu} = \mathbf{z} - \mathbf{H} \mathbf{x}_{k|k-1}$$

$$\mathbf{S} = \mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R}$$

$$\mathbf{W} = \mathbf{P}_{k|k-1} \mathbf{H}^T \mathbf{S}^{-1}$$

$$\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + \mathbf{W} \boldsymbol{\nu}$$

$$\mathbf{P}_{k|k} = \mathbf{P}_{k|k-1} - \mathbf{W} \mathbf{S} \mathbf{W}^T$$

Where:
- $\boldsymbol{\nu}$ is the measurement innovation (residual)
- $\mathbf{S}$ is the innovation covariance
- $\mathbf{W}$ is the Kalman gain
- $\mathbf{R}$ is the measurement noise covariance
- $\mathbf{H}$ is the measurement matrix

The Kalman gain $\mathbf{W}$ is the optimal weighting factor that minimizes estimation error, balancing prediction confidence against measurement confidence.

## State Representation

The filter maintains a 9-dimensional state vector representing position, velocity, and acceleration for each spatial axis:

$$\mathbf{x} = \begin{bmatrix} p_x \\ v_x \\ a_x \\ p_y \\ v_y \\ a_y \\ p_z \\ v_z \\ a_z \end{bmatrix}$$

Where:
- $p_i$ = position along axis $i$
- $v_i$ = velocity along axis $i$
- $a_i$ = acceleration along axis $i$ (assumed constant over $\Delta t$)

## Key Matrices

### State Transition Matrix (F)

Implements constant acceleration kinematics for each axis:

$$\mathbf{F}_{\text{block}} = \begin{bmatrix} 1 & \Delta t & \frac{1}{2}\Delta t^2 \\ 0 & 1 & \Delta t \\ 0 & 0 & 1 \end{bmatrix}$$

The full 9×9 matrix is block-diagonal with three identical blocks (one per axis):

$$\mathbf{F} = \begin{bmatrix} \mathbf{F}_{\text{block}} & \mathbf{0} & \mathbf{0} \\ \mathbf{0} & \mathbf{F}_{\text{block}} & \mathbf{0} \\ \mathbf{0} & \mathbf{0} & \mathbf{F}_{\text{block}} \end{bmatrix}$$

### Process Noise Covariance (D)

Models uncertainty in acceleration due to unknown dynamics:

$$\mathbf{D}_{\text{block}} = \sigma_a^2 \begin{bmatrix} \frac{\Delta t^4}{4} & \frac{\Delta t^3}{2} & \frac{\Delta t^2}{2} \\ \frac{\Delta t^3}{2} & \Delta t^2 & \Delta t \\ \frac{\Delta t^2}{2} & \Delta t & 1 \end{bmatrix}$$

With $\sigma_a = 0.2$ m/s² and block-diagonal structure.

### Measurement Matrix (H)

Maps the 9D state to the 2D Cartesian measurements (X and Y only):

$$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \end{bmatrix}$$

Note: Z position is not directly measured; it is estimated through the dynamics model.

### Initial Conditions

- Initial state: $\mathbf{x}_0 = \mathbf{0}$ (unknown starting condition)
- Initial covariance: $\mathbf{P}_0 = 1000 \cdot \mathbf{I}_9$ (large uncertainty)

## Assumptions and Limitations

### Key Assumptions

| Assumption | Description | Impact |
|-----------|-------------|--------|
| Linear Dynamics | System follows linear state-space model | Exact for constant acceleration; fails for maneuvers |
| Linear Measurements | Measurement matrix H is time-invariant and linear | Approximation; true radar measurements are nonlinear |
| Gaussian Noise | All noise sources are zero-mean Gaussian white noise | May not hold for sensor biases or colored noise |
| Constant Acceleration | Acceleration remains constant over $\Delta t$ | Smooth trajectories only; no sharp turns |
| 2D Observability | Only X and Y positions are directly measured | Z estimation relies entirely on dynamics; lacks altitude observability |
| Sequential Updates | Two radars updated sequentially in each time step | Suboptimal; joint update would reduce covariance |

### Limitations

1. **Nonlinear Approximation:** Polar-to-Cartesian conversion introduces approximation error. An Extended Kalman Filter (EKF) would handle this more rigorously.

2. **Z-Axis Unobservability:** Since altitude is not directly measured, the filter must infer it from the dynamics model. Poor process noise tuning can lead to poor Z-axis tracking.

3. **No Acceleration Observability:** Acceleration is not directly measured, only inferred. The filter may diverge if $\sigma_a$ is set too low.

4. **Fixed Geometry:** Radar positions are fixed and known. Real systems may have uncertain calibration.

5. **Time-Invariant Model:** Assumes the mountain pass follows the predefined sinusoidal model; real roads have varying dynamics.

## Usage

### Running the Simulation

```bash
python main.py
```

### Interactive Controls

The 3D visualization provides:
- **Next Step** button: Advance the simulation by one time step ($\Delta t = 2$ s)
- **Play Video** button: Automatically advance through the entire simulation
- **Checkboxes:** Toggle visibility of truth trajectory, radar measurements, and filter estimates

### Output

The simulation generates a 4-panel performance analysis:
1. **Total 3D Position Error:** Euclidean distance between truth and estimate over time
2. **X-Axis Tracking:** True vs. estimated X position
3. **Y-Axis Tracking:** True vs. estimated Y position
4. **Z-Axis Tracking:** True vs. estimated altitude (demonstrates observability limitations)

## File Structure

```
ISDF/
├── main.py                 # Main simulation loop and configuration
├── environment.py          # Ground truth trajectory model
├── sensors.py             # Radar sensor model
├── visualizer.py          # 3D visualization and UI
├── filters/
│   ├── base_filter.py     # Abstract filter base class
│   ├── kalman_filter.py   # Linear Kalman filter implementation
│   └── ekf.py             # Extended Kalman filter (optional)
└── README.md              # This file
```

## Further Work

Potential extensions to this framework:
- Implement an Extended Kalman Filter (EKF) to handle nonlinear measurements directly
- Add an Unscented Kalman Filter (UKF) for better nonlinear approximation
- Implement adaptive noise covariance tuning
- Add IMU measurements for improved Z-axis observability
- Compare filter performance metrics (RMS error, consistency checks)
