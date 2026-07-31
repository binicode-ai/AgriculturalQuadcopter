"""
navigation/waypoint.py

Waypoint navigation for autonomous flight.

Author: Biniyam Samuel
"""

from turtle import distance

import numpy as np


class WaypointNavigator:

    def __init__(self, waypoints, tolerance=0.5):

        self.waypoints = np.array(waypoints, dtype=float)

        self.current_index = 0

        self.tolerance = tolerance

    def current_waypoint(self):

        return self.waypoints[self.current_index]

    def update(self, position):

        target = self.current_waypoint()

        distance = np.linalg.norm(position - target)

        if distance < self.tolerance:

            if self.current_index < len(self.waypoints) - 1:

                self.current_index += 1

        return self.current_waypoint()


        def mission_complete(self, position):

         if self.current_index != len(self.waypoints) - 1:
            return False

        target = self.current_waypoint()

        distance = np.linalg.norm(position - target)

        return distance < self.tolerance