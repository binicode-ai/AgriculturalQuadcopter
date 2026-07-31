from controllers.pid import PID
import parameters as p
import numpy as np

class PositionController:

    def __init__(self):

        self.x_pid = PID(0.8,0,0.3)

        self.y_pid = PID(0.8,0,0.3)

        self.z_pid = PID(2.0,0.5,0.8)

    def update(

        self,

        desired_position,

        current_position,

        dt

    ):

        x_cmd = self.x_pid.update(
            desired_position[0],
            current_position[0],
            dt
        )

        y_cmd = self.y_pid.update(
            desired_position[1],
            current_position[1],
            dt
        )

        z_cmd = self.z_pid.update(
            desired_position[2],
            current_position[2],
            dt
        )

        desired_pitch = x_cmd / p.g

        desired_roll = -y_cmd / p.g

        # Limit commanded tilt angle

        max_angle = np.radians(20.0)

        desired_roll = np.clip(
        desired_roll,
        -max_angle,
        max_angle
    )

        desired_pitch = np.clip(
        desired_pitch,
        -max_angle,
            max_angle
)
        desired_thrust = (
            p.m*p.g
            +
            p.m*z_cmd
        )

        return (
            desired_roll,
            desired_pitch,
            desired_thrust
        )