"""
controller/attitude.py

Attitude controller using three PID controllers.

Author: Biniyam Samuel
"""

from controllers.pid import PID
import numpy as np


class AttitudeController:

    def __init__(self):

        # Conservative gains for initial tuning

        self.roll_pid = PID(
            kp=1.2,
            ki=0.0,
            kd=0.25
        )

        self.pitch_pid = PID(
            kp=1.2,
            ki=0.0,
            kd=0.25
        )

        self.yaw_pid = PID(
            kp=0.8,
            ki=0.0,
            kd=0.15
        )

    def update(
        self,
        desired_roll,
        desired_pitch,
        desired_yaw,
        estimated_roll,
        estimated_pitch,
        estimated_yaw,
        dt
    ):

        tau_roll = self.roll_pid.update(
            desired_roll,
            estimated_roll,
            dt
        )

        tau_pitch = self.pitch_pid.update(
            desired_pitch,
            estimated_pitch,
            dt
        )

        tau_yaw = self.yaw_pid.update(
            desired_yaw,
            estimated_yaw,
            dt
        )

        return (
            tau_roll,
            tau_pitch,
            tau_yaw
        )