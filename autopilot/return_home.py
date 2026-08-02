"""
autopilot/return_home.py

Return-To-Home controller.

Author: Biniyam Samuel
"""

import math


class ReturnHome:

    def __init__(self, arrival_radius=1.0):

        self.home = None

        self.arrival_radius = arrival_radius

        self.active = False

    # ----------------------------------------

    def set_home(self, x, y, z):

        self.home = (x, y, z)

        print(f"Home position set: {self.home}")

    # ----------------------------------------

    def activate(self):

        if self.home is None:

            raise RuntimeError("Home position has not been set.")

        self.active = True

        print("Return-To-Home activated.")

    # ----------------------------------------

    def deactivate(self):

        self.active = False

    # ----------------------------------------

    def is_active(self):

        return self.active

    # ----------------------------------------

    def target(self):

        return self.home

    # ----------------------------------------

    def distance_to_home(

        self,

        x,

        y,

        z

    ):

        hx, hy, hz = self.home

        return math.sqrt(

            (x - hx) ** 2 +

            (y - hy) ** 2 +

            (z - hz) ** 2

        )

    # ----------------------------------------

    def reached_home(

        self,

        x,

        y,

        z

    ):

        return (

            self.distance_to_home(

                x,

                y,

                z

            )

            <=

            self.arrival_radius

        )