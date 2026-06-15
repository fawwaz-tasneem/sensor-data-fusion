"""
Road map ComponentSpec.

The polygonal road map is constructed *from* the trajectory by sampling
it at uniform time intervals. The dashboard exposes:
  * n_nodes — sampling density (more nodes = less discretization error)
  * sigma_nodes — declared node-position uncertainty fed into the
    fictitious cross-track measurement variance

The actual node positions and arc-length array are computed by the
runner; there's no standalone `build` here because the road map
construction depends on the trajectory object.
"""
from __future__ import annotations

from sdf.scenarios import PolygonalRoadMap

from dashboard.schema import ComponentSpec, ParameterSpec


ROAD_MAP = ComponentSpec(
    label="Polygonal road map",
    constructor=PolygonalRoadMap,
    description=(
        "Piecewise-linear approximation of the trajectory. Enabling the road "
        "map turns on road-aided filtering: the road is fused as a fictitious "
        "cross-track measurement every step, which keeps the estimate on the "
        "road through sensor gaps such as a tunnel. It also positions the "
        "tunnel occlusion and draws the road in the scene."
    ),
    parameters=[
        ParameterSpec("n_nodes", int, default=30,
                      min=5, max=200, step=1,
                      description="Number of polygon nodes"),
        ParameterSpec("sigma_nodes", float, default=5.0,
                      min=0.1, max=50.0, step=0.1,
                      description="Declared node-position uncertainty",
                      unit="m"),
    ],
)
