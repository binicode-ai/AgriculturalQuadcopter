"""
mission/planner.py

High-level mission planner for agricultural surveys.

Author: Biniyam Samuel
"""

from mission.camera import Camera
from navigation.coverage import CoveragePlanner
from navigation.waypoint import WaypointNavigator


class MissionPlanner:

    def __init__(
        self,
        field_width,
        field_length,
        altitude,
        side_overlap,
        camera
    ):

        self.camera = camera

        spacing = camera.flight_spacing(
            altitude,
            side_overlap
        )

        planner = CoveragePlanner(
            field_width=field_width,
            field_length=field_length,
            line_spacing=spacing,
            altitude=altitude
        )

        self.waypoints = planner.generate()

        self.navigator = WaypointNavigator(
            self.waypoints
        )

    def current_target(self):

        return self.navigator.current_waypoint()

    def update(self, position):

        return self.navigator.update(position)

    def mission_complete(self, position):

        return self.navigator.mission_complete(position)