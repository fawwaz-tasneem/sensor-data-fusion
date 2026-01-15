import numpy as np
import matplotlib.pyplot as plt
from environment import MountainPassRoad
from sensors import RadarSensor
from filters.kalman_filter import KalmanFilter
from visualizer import FusionVisualizer

# --- Setup Parameters ---
dt = 2.0         # Time step Delta T = 2s
sigma_a = 0.2    # Process noise std dev (m/s^2)
t = 0.0 

# Radar Noise Parameters from requirements
sig_r = 10.0           # meters
sig_phi = np.deg2rad(0.1) # 0.1 degrees to radians

road = MountainPassRoad(dt)

# Dual Radars at specified coordinates (in meters)
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

# --- Initialization ---
x0 = np.zeros((9, 1)) 
P0 = np.eye(9) * 1000.0 # Large initial uncertainty
kf = KalmanFilter(x0, P0)

def simulation_step():
    global t
    # 1. Prediction
    kf.predict(F, D)
    gt = road.get_state(t)
    
    # 2. Get Raw Polar Measurements
    z1_polar = radar1.measure(gt) 
    z2_polar = radar2.measure(gt)
    
    # 3. CONVERSION: Polar -> Cartesian (Orchestrator Role)
    # Radar 1
    r1, p1 = z1_polar[0,0], z1_polar[1,0]
    rx1, ry1, rz1 = radar1.pos_s.flatten()
    z1_cart = np.array([[rx1 + r1*np.cos(p1)], [ry1 + r1*np.sin(p1)], [gt[2,0]]])
    
    # Radar 2
    r2, p2 = z2_polar[0,0], z2_polar[1,0]
    rx2, ry2, rz2 = radar2.pos_s.flatten()
    z2_cart = np.array([[rx2 + r2*np.cos(p2)], [ry2 + r2*np.sin(p2)], [gt[2,0]]])

    # 4. Noise Approximation (R_cart)
    # Simple diagonal approximation for Cartesian noise
    def get_R_cart(r, phi):
        var_x = (sig_r * np.cos(phi))**2 + (r * np.sin(phi) * sig_phi)**2
        var_y = (sig_r * np.sin(phi))**2 + (r * np.cos(phi) * sig_phi)**2
        return np.diag([var_x, var_y, 100.0]) # 100.0 as placeholder for z-variance

    R_cart1 = get_R_cart(r1, p1)
    R_cart2 = get_R_cart(r2, p2)

    # 5. Filter Update (Linear Update with Converted Measurements)
    kf.update(z1_cart, H_cart, R_cart1)
    kf.update(z2_cart, H_cart, R_cart2)
    
    # 6. Debugging
    print(f"Time: {t:.1f} | Truth: {gt[0,0]:.1f}, {gt[1,0]:.1f} | Est: {kf.x[0,0]:.1f}, {kf.x[3,0]:.1f}")

    # 7. Push to Visualizer
    filt_pos = kf.x[[0, 3, 6]].flatten()
    P_pos = kf.P[np.ix_([0, 3, 6], [0, 3, 6])]
    
    viz.update_data('truth', gt.flatten())
    viz.update_data('meas1', z1_cart.flatten())
    viz.update_data('meas2', z2_cart.flatten())
    viz.update_data('filt', filt_pos, cov=P_pos)
    
    t += dt

viz = FusionVisualizer(simulation_step)

if __name__ == "__main__":
    plt.show()