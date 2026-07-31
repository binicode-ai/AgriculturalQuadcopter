"""
agriculture/plant_tracker.py

Tracks sprayed plants.

Author: Biniyam Samuel
"""

import math


class PlantTracker:

    def __init__(

        self,

        distance_threshold=1.0

    ):

        self.distance_threshold = distance_threshold

        self.sprayed_plants = []

    # ------------------------------------------

    def already_sprayed(

        self,

        x,

        y

    ):

        for px, py in self.sprayed_plants:

            distance = math.sqrt(

                (x - px) ** 2 +

                (y - py) ** 2

            )

            if distance <= self.distance_threshold:

                return True

        return False

    # ------------------------------------------

    def add(

        self,

        x,

        y

    ):

        self.sprayed_plants.append(

            (x, y)

        )

    # ------------------------------------------

    def count(self):

        return len(

            self.sprayed_plants

        )

    # ------------------------------------------

    def clear(self):

        self.sprayed_plants.clear()