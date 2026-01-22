import numpy as np
import matplotlib.pyplot as plt
from environment import MountainPassRoad
from sensors import RadarSensor
from filters.kalman_filter import KalmanFilter
from visualizer import FusionVisualizer

# --- Setup Parameters ---
dt = 10.0         # Time step Delta T = 2s
sigma_a = 0.2    # Process noise standard deviation (m/s^2) (prediction noise)
t = 0.0
Z_AXIS_MEASURED = True  # Set to True to include elevation angle measurement 

# --- Stop Time Calculation ---
# 10km road at 20km/h
v_mps = 20 / 3.6  
dist_m = 10000    
T_stop = dist_m / v_mps

# Radar Noise Parameters from requirements
sig_r = 10.0           # meters
sig_phi = np.deg2rad(0.1) # 0.1 degrees converted to radians

road = MountainPassRoad(dt)

# Dual Radars at specified coordinates (in meters)
radar1 = RadarSensor(x_s=0, y_s=100000, z_s=10000, sigma_r=sig_r, sigma_phi=sig_phi)
radar2 = RadarSensor(x_s=100000, y_s=0, z_s=10000, sigma_r=sig_r, sigma_phi=sig_phi)

# --- Matrices: 9D Constant Acceleration ---
# ?? Check if this needs to be changed depending on the dynamics model
I = np.eye(3)
O = np.zeros((3,3))

# Transition Matrix F: pos = pos + v*dt + 0.5*a*dt^2 | v = v + a*dt
# ?? this probably needs to be changed depending on the dynamics model
f_block = np.array([[1, dt, 0.5*dt**2],
                    [0, 1,  dt],
                    [0, 0,  1]])
F = np.block([[f_block, O, O], [O, f_block, O], [O, O, f_block]])

# Process Noise Covariance D
# ?? This probably needs to be changed depending on the dynamics model
d_block = (sigma_a**2) * np.array([
    [dt**4/4, dt**3/2, dt**2/2],
    [dt**3/2, dt**2,   dt],
    [dt**2/2, dt,      1]
])
D = np.block([[d_block, O, O], [O, d_block, O], [O, O, d_block]])

# --- Measurement Matrix H ---
# Maps Cartesian measurements to the 9D state.
# If Z_AXIS_MEASURED: H is 3x9 (x, y, z measurements)
# Otherwise: H is 2x9 (x, y measurements only) (z axis is modeled in the state but not explciitly measured)

# ?? Check if this needs to be changed depending on the dynamics model (the model should only measure the postion,
# it should not get the velocity and the acceleration information)
if Z_AXIS_MEASURED:
    H_cart = np.zeros((3, 9))
    H_cart[0, 0] = 1.0 # px
    H_cart[1, 3] = 1.0 # py
    H_cart[2, 6] = 1.0 # pz
else:
    H_cart = np.zeros((2, 9))
    H_cart[0, 0] = 1.0 # px
    H_cart[1, 3] = 1.0 # py

# --- History Storage for Analysis ---
# This is alright, requires no changes
time_history = []
error_history = []
truth_x_history, truth_y_history, truth_z_history = [], [], []
filt_x_history, filt_y_history, filt_z_history = [], [], []

# --- Initialization ---
# ?? This is where an error might be happening. The state should be initialized by the first measurement, or 0. 
# I don't think that a 0 or identity matrix makes any sense. It should be initialized by the first measurement 
x0 = np.zeros((9, 1)) 
P0 = np.eye(9) * 1.0 # Large initial uncertainty
kf = KalmanFilter(x0, P0)

def simulation_step():
    global t

    # 1. Check Stop Condition
    if t >= T_stop:
        print(f"Reached end of road at {t:.1f}s. Stopping simulation.")
        viz.is_playing = False
        generate_performance_plots()
        return 
    
    # 2. Prediction
    kf.predict(F, D) # F and D is not initialized correctly maybe
    gt = road.get_state(t)  # returns ground truth data without adding any noise
    
    # 3. Sensor Measurement (Polar)
    z1_polar = radar1.measure(gt, include_elevation=Z_AXIS_MEASURED) 
    z2_polar = radar2.measure(gt, include_elevation=Z_AXIS_MEASURED)
    
    # 4. CONVERSION: Polar -> Cartesian
    # This should be done usign a helper function

    # No ground truth is fed here; we only use available sensor data.
    if Z_AXIS_MEASURED:
        # Polar: [range, azimuth, elevation]
        r1, p1, t1 = z1_polar[0,0], z1_polar[1,0], z1_polar[2,0]
        rx1, ry1, rz1 = radar1.pos_s.flatten()
        z1_cart = np.array([
            [rx1 + r1*np.cos(t1)*np.cos(p1)],
            [ry1 + r1*np.cos(t1)*np.sin(p1)],
            [rz1 + r1*np.sin(t1)]
        ])
        
        r2, p2, t2 = z2_polar[0,0], z2_polar[1,0], z2_polar[2,0]
        rx2, ry2, rz2 = radar2.pos_s.flatten()
        z2_cart = np.array([
            [rx2 + r2*np.cos(t2)*np.cos(p2)],
            [ry2 + r2*np.cos(t2)*np.sin(p2)],
            [rz2 + r2*np.sin(t2)]
        ])
    else:
        # Polar: [range, azimuth] only
        r1, p1 = z1_polar[0,0], z1_polar[1,0]
        rx1, ry1, rz1 = radar1.pos_s.flatten()
        z1_cart = np.array([[rx1 + r1*np.cos(p1)], [ry1 + r1*np.sin(p1)]])
        
        r2, p2 = z2_polar[0,0], z2_polar[1,0]
        rx2, ry2, rz2 = radar2.pos_s.flatten()
        z2_cart = np.array([[rx2 + r2*np.cos(p2)], [ry2 + r2*np.sin(p2)]])

    # 5. Noise Approximation (R_cart)
    if Z_AXIS_MEASURED:
        def get_R_cart_3d(r, phi, theta):
            # ?? Why a Jacobian approximation, this is not ekf. CHECK
            # Jacobian approximation for 3D polar to Cartesian conversion
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            cos_p = np.cos(phi)
            sin_p = np.sin(phi)
            
            var_x = (sig_r * cos_t * cos_p)**2 + (r * cos_t * sin_p * sig_phi)**2 + (r * (-sin_t) * cos_p * sig_phi)**2
            var_y = (sig_r * cos_t * sin_p)**2 + (r * cos_t * cos_p * sig_phi)**2 + (r * (-sin_t) * sin_p * sig_phi)**2
            var_z = (sig_r * sin_t)**2 + (r * cos_t * sig_phi)**2
            return np.diag([var_x, var_y, var_z])
        
        R_cart1 = get_R_cart_3d(r1, p1, t1)
        R_cart2 = get_R_cart_3d(r2, p2, t2)
    else:
        def get_R_cart_2d(r, phi):
            var_x = (sig_r * np.cos(phi))**2 + (r * np.sin(phi) * sig_phi)**2
            var_y = (sig_r * np.sin(phi))**2 + (r * np.cos(phi) * sig_phi)**2
            return np.diag([var_x, var_y])
        
        R_cart1 = get_R_cart_2d(r1, p1)
        R_cart2 = get_R_cart_2d(r2, p2)

    # 6. Filter Update (Sequential Fusion)
    # Change this to concurrent update (SEE SLIDES) 
    # The filter estimates pz based on the 9D state relations.
    kf.update(z1_cart, H_cart, R_cart1)
    kf.update(z2_cart, H_cart, R_cart2)
    
    # 7. Data Collection for Error Plots
    filt_pos = kf.x[[0, 3, 6]].flatten() # Extract px, py, and the ESTIMATED pz
    true_pos = gt.flatten()
    error = np.linalg.norm(true_pos - filt_pos)

    time_history.append(t); error_history.append(error)
    truth_x_history.append(true_pos[0]); filt_x_history.append(filt_pos[0])
    truth_y_history.append(true_pos[1]); filt_y_history.append(filt_pos[1])
    truth_z_history.append(true_pos[2]); filt_z_history.append(filt_pos[2])

    # 8. Push to Visualizer
    P_pos = kf.P[np.ix_([0, 3, 6], [0, 3, 6])]
    viz.update_data('truth', true_pos)
    
    # Sending 3D lists to visualizer
    if Z_AXIS_MEASURED:
        # Measurements include Z
        viz.update_data('meas1', [z1_cart[0,0], z1_cart[1,0], z1_cart[2,0]]) 
        viz.update_data('meas2', [z2_cart[0,0], z2_cart[1,0], z2_cart[2,0]])
    else:
        # Measurements are plotted at Z=0 (floor level) since height is not measured.
        viz.update_data('meas1', [z1_cart[0,0], z1_cart[1,0], 0]) 
        viz.update_data('meas2', [z2_cart[0,0], z2_cart[1,0], 0])
    
    # The filter estimate (filt_pos) includes the 3D estimated Z value
    viz.update_data('filt', filt_pos, cov=P_pos)
    
    t += dt

# --- Initialization of Visualizer ---
viz = FusionVisualizer(simulation_step)
viz.set_radar_positions(radar1.pos_s, radar2.pos_s)

def generate_performance_plots():
    """Generates a 5-panel performance analysis including a 2D Top-Down View"""
    # Create a figure with a custom grid
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 2)
    
    # --- Top-Down View (Spans two rows on the right) ---
    ax_topdown = fig.add_subplot(gs[0:2, 1])
    ax_topdown.plot(truth_x_history, truth_y_history, 'g-', label='Ground Truth Path', alpha=0.7)
    ax_topdown.plot(filt_x_history, filt_y_history, 'b--', label='KF Estimated Path', alpha=0.8)
    # Mark Radar Positions
    '''
    ax_topdown.scatter([radar1.pos_s[0], radar2.pos_s[0]], 
                       [radar1.pos_s[1], radar2.pos_s[1]], 
                       c='red', marker='^', label='Radars')
    '''
    ax_topdown.set_title("2D Top-Down View (XY Plane)")
    ax_topdown.set_xlabel("X Position (m)")
    ax_topdown.set_ylabel("Y Position (m)")
    ax_topdown.legend()
    ax_topdown.grid(True)
    ax_topdown.set_aspect('equal', adjustable='datalim')

    # --- Plot 1: Total 3D Error (Top Left) ---
    ax_err = fig.add_subplot(gs[0, 0])
    ax_err.plot(time_history, error_history, 'r-', label='3D Position Error')
    ax_err.set_title("Filter Performance: Total Positioning Error")
    ax_err.set_ylabel("Error (m)")
    ax_err.grid(True)

    # --- Plot 2: X-Axis Tracking (Middle Left) ---
    ax_x = fig.add_subplot(gs[1, 0], sharex=ax_err)
    ax_x.plot(time_history, truth_x_history, 'g-', label='Truth X')
    ax_x.plot(time_history, filt_x_history, 'b--', label='Est X')
    ax_x.set_title("X-Axis Tracking")
    ax_x.grid(True)

    # --- Plot 3: Y-Axis Tracking (Bottom Left) ---
    ax_y = fig.add_subplot(gs[2, 0], sharex=ax_err)
    ax_y.plot(time_history, truth_y_history, 'g-', label='Truth Y')
    ax_y.plot(time_history, filt_y_history, 'b--', label='Est Y')
    ax_y.set_title("Y-Axis Tracking")
    ax_y.grid(True)

    # --- Plot 4: Z Tracking (Bottom Span) ---
    ax_z = fig.add_subplot(gs[3, :])
    ax_z.plot(time_history, truth_z_history, 'g-', label='Truth Z')
    ax_z.plot(time_history, filt_z_history, 'b--', label='Est Z')
    ax_z.set_title("Z-Axis (Altitude) Estimation")
    ax_z.set_xlabel("Time (s)")
    ax_z.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Simulation started. Use the UI buttons to Step or Play.")
    plt.show()