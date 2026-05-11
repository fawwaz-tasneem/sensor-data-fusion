"""
Visualization helpers for the SDF framework.

This package contains reusable plot functions kept separate from any
particular example. Examples and the Dash dashboard both import from
here so that, e.g., the tunnel wireframe rendering is defined once.

Each helper takes a matplotlib axis (or Plotly figure) plus the world
object to draw, and adds to it without claiming the figure. The caller
owns the figure layout.
"""
from sdf.viz.terrain_mesh import draw_tunnel_wireframe, tunnel_wireframe_segments

__all__ = ["draw_tunnel_wireframe", "tunnel_wireframe_segments"]
