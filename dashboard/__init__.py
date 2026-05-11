"""
SDF tracking framework dashboard.

A Plotly Dash app that lets a user configure a tracking scenario
(trajectory, sensors, occlusion, filter, road map) through a form-based
UI, then runs the scenario and shows an animated 3D playback plus
error/altitude/detection panels. The result can be exported as an MP4.

Run with:
    python -m dashboard

The dashboard sits outside the `sdf` package so the framework itself
doesn't depend on Dash/Plotly.
"""
