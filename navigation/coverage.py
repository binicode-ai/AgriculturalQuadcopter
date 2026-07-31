"""
navigation/coverage.py

Generate lawnmower coverage waypoints.

Author: Biniyam Samuel
"""

import numpy as np


class CoveragePlanner:

    def __init__(self,
                 field_width,
                 field_length,
                 line_spacing,
                 altitude):

        self.width = field_width
        self.length = field_length
        self.spacing = line_spacing
        self.altitude = altitude

    def generate(self):

        waypoints = []

        x_positions = np.arange(
            0,
            self.width + self.spacing,
            self.spacing
        )

        for i, x in enumerate(x_positions):

            if i % 2 == 0:

                # Bottom to top
                waypoints.append([x, 0, self.altitude])
                waypoints.append([x, self.length, self.altitude])

            else:

                # Top to bottom
                waypoints.append([x, self.length, self.altitude])
                waypoints.append([x, 0, self.altitude])

        return np.array(waypoints)