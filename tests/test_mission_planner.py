import numpy as np

from mission.camera import Camera
from mission.planner import MissionPlanner


camera = Camera(
    sensor_width=13.2,
    sensor_height=8.8,
    focal_length=8.8,
    image_width=5472,
    image_height=3648
)

mission = MissionPlanner(
    field_width=100,
    field_length=60,
    altitude=40,
    side_overlap=0.70,
    camera=camera
)

print("Mission Waypoints:\n")

for i, wp in enumerate(mission.waypoints):
    print(f"{i+1:2d}: {wp}")

print("\nCurrent Target:")
print(mission.current_target())

position = np.array([0.0, 0.0, 40.0])

mission.update(position)

print("\nNext Target:")
print(mission.current_target())