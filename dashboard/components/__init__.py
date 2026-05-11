"""
Component registries for the dashboard.

Each module in this package defines one or more `ComponentSpec` entries
plus a `ComponentChoice` that groups specs of the same kind (all
trajectories, all sensors, etc.). The UI and the simulation runner
import from here.
"""
from dashboard.components.filters import FILTER_CHOICE
from dashboard.components.motion_models import MOTION_MODEL_CHOICE
from dashboard.components.occlusion import OCCLUSION_CHOICE
from dashboard.components.platforms import PLATFORM_CHOICE
from dashboard.components.road_map import ROAD_MAP
from dashboard.components.sensor_list import SENSOR_LIST, SensorListSpec
from dashboard.components.sensors import SENSOR_CHOICE
from dashboard.components.trajectories import TRAJECTORY_CHOICE

__all__ = [
    "TRAJECTORY_CHOICE",
    "MOTION_MODEL_CHOICE",
    "SENSOR_CHOICE",
    "SENSOR_LIST",
    "SensorListSpec",
    "OCCLUSION_CHOICE",
    "PLATFORM_CHOICE",
    "ROAD_MAP",
    "FILTER_CHOICE",
]
