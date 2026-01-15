import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

class FusionVisualizer:
    def __init__(self, step_callback):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.step_callback = step_callback
        self.is_playing = False

        # Independent Storage - Add both measurement keys here
        self.data = {
            'truth': {'x': [], 'y': [], 'z': []},
            'meas1': {'x': [], 'y': [], 'z': []},  # Added
            'meas2': {'x': [], 'y': [], 'z': []},  # Added
            'filt':  {'x': [], 'y': [], 'z': []}
        }

        # Initialize Line Objects - Assign different colors
        self.lines = {
            'truth': self.ax.plot([], [], [], 'g-', label='Truth', alpha=0.7)[0],
            'meas1': self.ax.plot([], [], [], 'r.', label='Radar 1', markersize=3)[0],
            'meas2': self.ax.plot([], [], [], 'm.', label='Radar 2', markersize=3)[0],
            'filt':  self.ax.plot([], [], [], 'b--', label='Filter')[0]
        }

        # Set initial environment scale
        self.ax.set_xlim(0, 10000)
        self.ax.set_ylim(-2000, 2000)
        self.ax.set_zlim(-1500, 1500)
        self.ax.legend()

        # UI Buttons
        ax_step = plt.axes([0.7, 0.02, 0.1, 0.05])
        ax_play = plt.axes([0.81, 0.02, 0.1, 0.05])
        self.btn_step = Button(ax_step, 'Next Step')
        self.btn_play = Button(ax_play, 'Play Video')
        
        self.btn_step.on_clicked(lambda e: self.step_callback())
        self.btn_play.on_clicked(self._toggle_play)

    def _toggle_play(self, event):
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text('Stop' if self.is_playing else 'Play Video')
        if self.is_playing:
            self._play_loop()

    def _play_loop(self):
        while self.is_playing:
            self.step_callback()
            plt.pause(0.01) # Keeps GUI responsive and maintains zoom

    import numpy as np

# ... inside FusionVisualizer class ...

    def update_data(self, key, pos, cov=None):
        """
        Agnostic update: receives a key and a list [x, y, z].
        Now also accepts an optional 3x3 covariance matrix 'cov'.
        """
        self.data[key]['x'].append(pos[0])
        self.data[key]['y'].append(pos[1])
        self.data[key]['z'].append(pos[2])
        
        # Update the line data
        self.lines[key].set_data(self.data[key]['x'], self.data[key]['y'])
        self.lines[key].set_3d_properties(self.data[key]['z'])
        
        # Draw uncertainty ellipsoid if covariance is provided and it's the filter
        if key == 'filt' and cov is not None:
            self._draw_ellipsoid(pos, cov)
            
        self.fig.canvas.draw_idle()

    def _draw_ellipsoid(self, pos, cov):
        """
        Calculates and renders a 3D wireframe ellipsoid based on the P matrix.
        """
        # 1. Get eigenvalues and eigenvectors of the 3x3 position covariance
        vals, vecs = np.linalg.eigh(cov)
        
        # 2. Calculate radii (2-sigma for ~95% confidence)
        # Max(0, val) prevents math errors if P becomes non-positive definite
        radii = 2 * np.sqrt(np.maximum(vals, 0))
        
        # 3. Create a unit sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        
        # 4. Transform sphere into ellipsoid
        for i in range(len(u)):
            for j in range(len(v)):
                # Rotate and scale the point
                point = np.array([x[i,j], y[i,j], z[i,j]])
                transformed = vecs @ (point * radii) + pos
                x[i,j], y[i,j], z[i,j] = transformed
        
        # 5. Plot the wireframe (only one at a time to avoid clutter)
        # Check if an old ellipsoid exists and remove it to keep the view clean
        if hasattr(self, 'current_ellipsoid'):
            self.current_ellipsoid.remove()
        
        self.current_ellipsoid = self.ax.plot_wireframe(x, y, z, color='b', alpha=0.2, linewidth=0.5)