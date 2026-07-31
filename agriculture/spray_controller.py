"""
agriculture/spray_controller.py

Controls the agricultural sprayer.

Author: Biniyam Samuel
"""

import time


class SprayController:

    def __init__(

        self,

        tank_capacity=5.0,      # liters

        flow_rate=0.05,         # liters/second

        spray_duration=1.0,     # seconds

        cooldown=0.5            # seconds

    ):

        self.tank_capacity = tank_capacity
        self.remaining = tank_capacity

        self.flow_rate = flow_rate

        self.spray_duration = spray_duration

        self.cooldown = cooldown

        self.last_spray_time = -1e9

    # -------------------------------------------------

    def can_spray(self):

        if self.remaining <= 0:

            return False

        elapsed = time.time() - self.last_spray_time

        return elapsed >= self.cooldown

    # -------------------------------------------------

    def spray(self):

        if not self.can_spray():

            print("Sprayer unavailable.")

            return False

        used = self.flow_rate * self.spray_duration

        if used > self.remaining:

            used = self.remaining

        self.remaining -= used

        self.last_spray_time = time.time()

        print()

        print("===== SPRAY =====")

        print(f"Duration : {self.spray_duration:.2f} s")

        print(f"Used     : {used:.3f} L")

        print(f"Remaining: {self.remaining:.3f} L")

        print("=================")

        return True

    # -------------------------------------------------

    def refill(self):

        self.remaining = self.tank_capacity

        print("Tank refilled.")

    # -------------------------------------------------

    def level_percent(self):

        return 100.0 * self.remaining / self.tank_capacity