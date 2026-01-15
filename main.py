import numpy as np
import matplotlib.pyplot as plt
from environment import MountainPassRoad
from sensors import RadarSensor
from filters.kalman_filter import KalmanFilter
from visualizer import FusionVisualizer

# --- Setup Parameters ---
dt = 2.0 
sigma_a = 0.2
t = 0.0
road = MountainPassRoad(dt)
radar = RadarSensor(x_s=0, y_s=100000, z_s=10000)

# --- Matrices: Constant Acceleration ---
I = np.eye(3)
O = np.zeros((3,3))
F = np.block([[I, dt*I], [O, I]])

d11, d12, d22 = (dt**4/4)*I, (dt**3/2)*I, (dt**2)*I
D = (sigma_a**2) * np.block([[d11, d12], [d12, d22]])

# --- FAILURE SETUP: Naive H Matrix ---
# Mapping 2 radar values to the first 2 state variables
H = np.zeros((2, 6))
H[0, 0] = 1.0 
H[1, 1] = 1.0 

# --- Initialization ---
x0 = np.zeros((6, 1))
P0 = np.eye(6) * 1000.0 
kf = KalmanFilter(x0, P0)


def simulation_step():
    global t
    # 1. Prediction & True State
    kf.predict(F, D)
    gt = road.get_state(t)
    
    # 2. Sensor measurement (Nonlinear)
    z = radar.measure(gt) 
    
    # 3. Filter Update (Naive Failure)
    kf.update(z, H, radar.R)
    
    # 4. TRANSLATION: Convert everything to [x, y, z] for the Agnostic Visualizer
    # Translate Measurement
    rx, ry, rz = radar.pos_s.flatten()
    rng, phi = z[0,0], z[1,0]
    obs_x = rx + rng * np.cos(phi)
    obs_y = ry + rng * np.sin(phi)
    obs_z = 0 # Azimuth only radar assumption

    # Translate Filter Estimate (just taking px, py, pz)
    filt_pos = kf.x[:3].flatten()

    # 5. Push to Visualizer
    viz.update_data('truth', gt.flatten())
    viz.update_data('meas', [obs_x, obs_y, obs_z])
    viz.update_data('filt', filt_pos)
    
    t += dt
    
# 2. PASS THE FUNCTION TO THE VISUALIZER
viz = FusionVisualizer(simulation_step)

if __name__ == "__main__":
    print("Click 'Next Step' or 'Play Video' in the plot window.")
    plt.show()