"""
controllers/flight_manager.py
"""

from controllers.position import PositionController
from controllers.attitude import AttitudeController
from controllers.mixer import mix


class FlightManager:

    def __init__(self):

        self.position = PositionController()

        self.attitude = AttitudeController()

    def update(

        self,

        desired_position,

        estimated_state,

        dt

    ):

        phi = estimated_state[3]
        theta = estimated_state[4]
        psi = estimated_state[5]

        desired_roll, desired_pitch, desired_thrust = (

            self.position.update(

                desired_position,

                estimated_state[:3],

                dt

            )

        )

        desired_yaw = 0.0

        tau_roll, tau_pitch, tau_yaw = (

            self.attitude.update(

                desired_roll,

                desired_pitch,

                desired_yaw,

                phi,

                theta,

                psi,

                dt

            )

        )

        motor_speed = mix(

            desired_thrust,

            tau_roll,

            tau_pitch,

            tau_yaw

        )

        return motor_speed