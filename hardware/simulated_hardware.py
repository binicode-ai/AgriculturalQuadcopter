"""
hardware/simulated_hardware.py

Simulation hardware backend.
"""

from hardware.hardware_interface import HardwareInterface


class SimulatedHardware(HardwareInterface):

    def __init__(self):

        self.armed = False

    def arm(self):

        self.armed = True

        print("Simulation armed.")

    def disarm(self):

        self.armed = False

        print("Simulation disarmed.")

    def set_motor_speeds(self, speeds):

        print("Motor Speeds:", speeds)

    def read_gps(self):

        return {

            "x": 0.0,

            "y": 0.0,

            "z": 2.0

        }

    def read_imu(self):

        return {

            "roll": 0.0,

            "pitch": 0.0,

            "yaw": 0.0

        }

    def capture_image(self):

        print("Captured simulated image.")

        return "data/sample_field.jpg"

    def spray(self):

        print("Simulated spraying.")