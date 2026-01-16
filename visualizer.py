import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
import numpy as np

class FusionVisualizer:
    def __init__(self, step_callback):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.step_callback = step_callback
        self.is_playing = False

        # Independent Storage
        self.data = {
            'truth': {'x': [], 'y': [], 'z': []},
            'meas1': {'x': [], 'y': [], 'z': []},
            'meas2': {'x': [], 'y': [], 'z': []},
            'filt':  {'x': [], 'y': [], 'z': []}
        }

        # Initialize Line Objects
        self.lines = {
            'truth': self.ax.plot([], [], [], 'g-', label='Truth', alpha=0.7)[0],
            'meas1': self.ax.plot([], [], [], 'r.', label='Radar 1', markersize=3)[0],
            'meas2': self.ax.plot([], [], [], 'm.', label='Radar 2', markersize=3)[0],
            'filt':  self.ax.plot([], [], [], 'b--', label='Filter')[0]
        }

        # Environment scale - Locked to Road
        self.ax.set_xlim(0, 10000)
        self.ax.set_ylim(-2000, 2000)
        self.ax.set_zlim(-1500, 1500)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.legend(loc='upper left')

        # --- UI Elements ---
        ax_step = plt.axes([0.7, 0.02, 0.1, 0.05])
        ax_play = plt.axes([0.81, 0.02, 0.1, 0.05])
        self.btn_step = Button(ax_step, 'Next Step')
        self.btn_play = Button(ax_play, 'Play Video')
        
        self.btn_step.on_clicked(lambda e: self.step_callback())
        self.btn_play.on_clicked(self._toggle_play)

        # Checkboxes for Visibility
        rax = plt.axes([0.02, 0.4, 0.12, 0.15])
        self.labels = ['Truth', 'Radar 1', 'Radar 2', 'Filter']
        self.visibility = [True, True, True, True]
        self.check = CheckButtons(rax, self.labels, self.visibility)
        self.check.on_clicked(self._on_checkbox_click)

    def set_radar_positions(self, pos1, pos2):
        """Plots the static locations of the radar towers"""
        # Plot Radar 1
        self.ax.scatter(pos1[0], pos1[1], pos1[2], c='red', marker='^', s=100)
        self.ax.text(pos1[0], pos1[1], pos1[2], " Radar 1", color='red')
        
        # Plot Radar 2
        self.ax.scatter(pos2[0], pos2[1], pos2[2], c='magenta', marker='^', s=100)
        self.ax.text(pos2[0], pos2[1], pos2[2], " Radar 2", color='magenta')
        
        # Re-verify limits so the view stays on the car, not the distant radars
        self.ax.set_xlim(0, 10000)
        self.ax.set_ylim(-2000, 2000)
        self.ax.set_zlim(-1500, 1500)
        self.fig.canvas.draw_idle()

    def _on_checkbox_click(self, label):
        mapping = {'Truth': 'truth', 'Radar 1': 'meas1', 'Radar 2': 'meas2', 'Filter': 'filt'}
        key = mapping[label]
        curr_vis = self.lines[key].get_visible()
        self.lines[key].set_visible(not curr_vis)
        
        if key == 'filt' and hasattr(self, 'current_ellipsoid'):
            self.current_ellipsoid.set_visible(not curr_vis)
        self.fig.canvas.draw_idle()

    def _toggle_play(self, event):
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text('Stop' if self.is_playing else 'Play Video')
        if self.is_playing:
            self._play_loop()

    def _play_loop(self):
        while self.is_playing:
            self.step_callback()
            plt.pause(0.01)

    def update_data(self, key, pos, cov=None):
        self.data[key]['x'].append(pos[0])
        self.data[key]['y'].append(pos[1])
        self.data[key]['z'].append(pos[2])
        self.lines[key].set_data(self.data[key]['x'], self.data[key]['y'])
        self.lines[key].set_3d_properties(self.data[key]['z'])
        
        if key == 'filt' and cov is not None:
            self._draw_ellipsoid(pos, cov)
        self.fig.canvas.draw_idle()

    def _draw_ellipsoid(self, pos, cov):
            vals, vecs = np.linalg.eigh(cov)
            radii = 2 * np.sqrt(np.maximum(vals, 0))
            
            # 3. Create a unit sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones_like(u), np.cos(v))
            
            # FIX: Use the shape of the resulting 2D arrays (x, y, or z) 
            # instead of the length of vectors u or v.
            rows, cols = x.shape 
            for i in range(rows):
                for j in range(cols):
                    point = np.array([x[i,j], y[i,j], z[i,j]])
                    # Apply rotation (vecs) and scaling (radii)
                    transformed = vecs @ (point * radii) + pos
                    x[i,j], y[i,j], z[i,j] = transformed
            
            if hasattr(self, 'current_ellipsoid'):
                self.current_ellipsoid.remove()
            
            self.current_ellipsoid = self.ax.plot_wireframe(x, y, z, color='b', alpha=0.2, linewidth=0.5)
            self.current_ellipsoid.set_visible(self.lines['filt'].get_visible())