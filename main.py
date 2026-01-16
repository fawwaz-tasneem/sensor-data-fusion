import numpy as np
import matplotlib.pyplot as plt
from environment import MountainPassRoad
from sensors import RadarSensor
from filters.kalman_filter import KalmanFilter
from visualizer import FusionVisualizer

# --- Setup Parameters ---
dt = 2.0         # Time step Delta T = 2s
sigma_a = 0.2    # Process noise standard deviation (m/s^2)
t = 0.0 

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
# Radar 1: (0, 100km, 10km) | Radar 2: (100km, 0, 10km)
radar1 = RadarSensor(x_s=0, y_s=100000, z_s=10000)
radar2 = RadarSensor(x_s=100000, y_s=0, z_s=10000)

# --- Matrices: 9D Constant Acceleration ---
I = np.eye(3)
O = np.zeros((3,3))

# Transition Matrix F: pos = pos + v*dt + 0.5*a*dt^2 | v = v + a*dt
f_block = np.array([[1, dt, 0.5*dt**2],
                    [0, 1,  dt],
                    [0, 0,  1]])
F = np.block([[f_block, O, O], [O, f_block, O], [O, O, f_block]])

# Process Noise Covariance D
d_block = (sigma_a**2) * np.array([
    [dt**4/4, dt**3/2, dt**2/2],
    [dt**3/2, dt**2,   dt],
    [dt**2/2, dt,      1]
])
D = np.block([[d_block, O, O], [O, d_block, O], [O, O, d_block]])

# --- Measurement Matrix H ---
# Maps 3D Cartesian [x, y, z] measurements to 9D state [px, vx, ax, py, vy, ay, pz, vz, az]
H_cart = np.zeros((3, 9))
H_cart[0, 0] = 1.0 # px
H_cart[1, 3] = 1.0 # py
H_cart[2, 6] = 1.0 # pz

# --- History Storage for Analysis ---
time_history = []
error_history = []
truth_x_history, truth_y_history, truth_z_history = [], [], []
filt_x_history, filt_y_history, filt_z_history = [], [], []

# --- Initialization ---
x0 = np.zeros((9, 1)) 
P0 = np.eye(9) * 1000.0 # Large initial uncertainty
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
    kf.predict(F, D)
    gt = road.get_state(t)
    
    # 3. Sensor Measurement (Polar)
    z1_polar = radar1.measure(gt) 
    z2_polar = radar2.measure(gt)
    
    # 4. CONVERSION: Polar -> Cartesian (Orchestrator Role)
    # Radar 1
    r1, p1 = z1_polar[0,0], z1_polar[1,0]
    rx1, ry1, rz1 = radar1.pos_s.flatten()
    z1_cart = np.array([[rx1 + r1*np.cos(p1)], [ry1 + r1*np.sin(p1)], [gt[2,0]]])
    
    # Radar 2
    r2, p2 = z2_polar[0,0], z2_polar[1,0]
    rx2, ry2, rz2 = radar2.pos_s.flatten()
    z2_cart = np.array([[rx2 + r2*np.cos(p2)], [ry2 + r2*np.sin(p2)], [gt[2,0]]])

    # 5. Noise Approximation (R_cart)
    def get_R_cart(r, phi):
        var_x = (sig_r * np.cos(phi))**2 + (r * np.sin(phi) * sig_phi)**2
        var_y = (sig_r * np.sin(phi))**2 + (r * np.cos(phi) * sig_phi)**2
        return np.diag([var_x, var_y, 100.0]) # 100.0 placeholder for missing elevation z-variance

    R_cart1 = get_R_cart(r1, p1)
    R_cart2 = get_R_cart(r2, p2)

    # 6. Filter Update (Sequential Fusion)
    kf.update(z1_cart, H_cart, R_cart1)
    kf.update(z2_cart, H_cart, R_cart2)
    
    # 7. Data Collection for Error Plots
    filt_pos = kf.x[[0, 3, 6]].flatten()
    true_pos = gt.flatten()
    error = np.linalg.norm(true_pos - filt_pos)

    time_history.append(t)
    error_history.append(error)
    truth_x_history.append(true_pos[0]); filt_x_history.append(filt_pos[0])
    truth_y_history.append(true_pos[1]); filt_y_history.append(filt_pos[1])
    truth_z_history.append(true_pos[2]); filt_z_history.append(filt_pos[2])

    # 8. Push to Visualizer
    P_pos = kf.P[np.ix_([0, 3, 6], [0, 3, 6])]
    viz.update_data('truth', true_pos)
    viz.update_data('meas1', z1_cart.flatten())
    viz.update_data('meas2', z2_cart.flatten())
    viz.update_data('filt', filt_pos, cov=P_pos)
    
    t += dt

# --- Initialization of Visualizer ---
viz = FusionVisualizer(simulation_step)
viz.set_radar_positions(radar1.pos_s, radar2.pos_s) # Plot static radar towers

def generate_performance_plots():
    """Generates the 4-panel performance analysis"""
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: Total 3D Error
    axes[0].plot(time_history, error_history, 'r-', label='3D Position Error')
    axes[0].set_title("Filter Performance: Total Positioning Error")
    axes[0].set_ylabel("Error (m)")
    axes[0].grid(True)
    
    # Plot 2, 3, 4: Axis Tracking Comparison
    axes[1].plot(time_history, truth_x_history, 'g-', label='Truth X')
    axes[1].plot(time_history, filt_x_history, 'b--', label='Est X')
    axes[1].set_title("X-Axis Convergence")
    
    axes[2].plot(time_history, truth_y_history, 'g-', label='Truth Y')
    axes[2].plot(time_history, filt_y_history, 'b--', label='Est Y')
    axes[2].set_title("Y-Axis Convergence")
    
    axes[3].plot(time_history, truth_z_history, 'g-', label='Truth Z')
    axes[3].plot(time_history, filt_z_history, 'b--', label='Est Z')
    axes[3].set_title("Z-Axis (Altitude) Convergence")
    axes[3].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Simulation started. Use the UI buttons to Step or Play.")
    plt.show()