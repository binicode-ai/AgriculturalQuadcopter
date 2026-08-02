"""
hardware/real_hardware.py

Placeholder for real drone hardware.

Author: Biniyam Samuel
"""

from hardware.hardware_interface import HardwareInterface


class RealHardware(HardwareInterface):

    def arm(self):
        raise NotImplementedError

    def disarm(self):
        raise NotImplementedError

    def set_motor_speeds(self, speeds):
        raise NotImplementedError

    def read_gps(self):
        raise NotImplementedError

    def read_imu(self):
        raise NotImplementedError

    def capture_image(self):
        raise NotImplementedError

    def spray(self):
        raise NotImplementedError