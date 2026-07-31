"""
state_estimator.py

Maintains estimated attitude using the
Complementary Filter.
"""

import parameters as p

from estimation.complementary import (
    estimate_roll,
    estimate_pitch
)


class StateEstimator:

    def __init__(self):

        self.roll = 0.0
        self.pitch = 0.0

    def update(
        self,
        gyro,
        accel
    ):

        self.roll = estimate_roll(

            self.roll,

            gyro[0],

            accel[1],

            accel[2],

            p.dt

        )

        self.pitch = estimate_pitch(

            self.pitch,

            gyro[1],

            accel[0],

            accel[1],

            accel[2],

            p.dt

        )

        return self.roll, self.pitch