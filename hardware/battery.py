"""
hardware/battery.py

Battery Management System.

Author: Biniyam Samuel
"""


class Battery:

    def __init__(

        self,

        capacity_wh=220.0,

        voltage=22.2,

        low_level=20.0,

        critical_level=10.0

    ):

        self.capacity_wh = capacity_wh

        self.remaining_wh = capacity_wh

        self.voltage = voltage

        self.low_level = low_level

        self.critical_level = critical_level

    # -------------------------------------

    def consume(

        self,

        power_watts,

        dt

    ):

        energy = (

            power_watts *

            dt /

            3600.0

        )

        self.remaining_wh = max(

            0.0,

            self.remaining_wh - energy

        )

    # -------------------------------------

    def percentage(self):

        return (

            100 *

            self.remaining_wh /

            self.capacity_wh

        )

    # -------------------------------------

    def is_low(self):

        return (

            self.percentage()

            <= self.low_level

        )

    # -------------------------------------

    def is_critical(self):

        return (

            self.percentage()

            <= self.critical_level

        )

    # -------------------------------------

    def estimated_time(

        self,

        average_power

    ):

        if average_power <= 0:

            return float("inf")

        hours = (

            self.remaining_wh /

            average_power

        )

        return hours * 60