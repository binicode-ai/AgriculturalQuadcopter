"""
hardware/hardware_interface.py

Abstract hardware interface.

Author: Biniyam Samuel
"""

from abc import ABC, abstractmethod


class HardwareInterface(ABC):

    @abstractmethod
    def arm(self):
        pass

    @abstractmethod
    def disarm(self):
        pass

    @abstractmethod
    def set_motor_speeds(self, speeds):
        pass

    @abstractmethod
    def read_gps(self):
        pass

    @abstractmethod
    def read_imu(self):
        pass

    @abstractmethod
    def capture_image(self):
        pass

    @abstractmethod
    def spray(self):
        pass