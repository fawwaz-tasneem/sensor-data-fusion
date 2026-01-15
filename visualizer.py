import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

class FusionVisualizer:
    def __init__(self, step_callback):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.step_callback = step_callback
        self.is_playing = False

        # Independent Storage
        self.data = {
            'truth': {'x': [], 'y': [], 'z': []},
            'meas':  {'x': [], 'y': [], 'z': []},
            'filt':  {'x': [], 'y': [], 'z': []}
        }

        # Initialize Line Objects (This allows us to update data without clearing the axis)
        self.lines = {
            'truth': self.ax.plot([], [], [], 'g-', label='Truth', alpha=0.7)[0],
            'meas':  self.ax.plot([], [], [], 'r.', label='Obs', markersize=3)[0],
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

    def update_data(self, key, pos):
        """Purely agnostic: receives a key and a list [x, y, z]"""
        self.data[key]['x'].append(pos[0])
        self.data[key]['y'].append(pos[1])
        self.data[key]['z'].append(pos[2])
        
        # Update the specific line object without clearing the axis
        self.lines[key].set_data(self.data[key]['x'], self.data[key]['y'])
        self.lines[key].set_3d_properties(self.data[key]['z'])
        
        self.fig.canvas.draw_idle() # Redraw only if necessary