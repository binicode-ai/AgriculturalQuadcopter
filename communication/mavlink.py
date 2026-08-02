"""
communication/mavlink.py

MAVLink communication layer.

Author: Biniyam Samuel
"""

from pymavlink import mavutil


class MAVLinkInterface:

    def __init__(

        self,

        connection_string="udp:127.0.0.1:14550"

    ):

        self.connection = mavutil.mavlink_connection(

            connection_string

        )

    # --------------------------------------------

    def wait_heartbeat(self):

        print("Waiting for heartbeat...")

        self.connection.wait_heartbeat()

        print("Heartbeat received!")

    # --------------------------------------------

    def arm(self):

        self.connection.mav.command_long_send(

            self.connection.target_system,

            self.connection.target_component,

            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,

            0,

            1,

            0,

            0,

            0,

            0,

            0,

            0

        )

        print("Arm command sent.")

    # --------------------------------------------

    def disarm(self):

        self.connection.mav.command_long_send(

            self.connection.target_system,

            self.connection.target_component,

            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,

            0,

            0,

            0,

            0,

            0,

            0,

            0,

            0

        )

        print("Disarm command sent.")

    # --------------------------------------------

    def read_gps(self):

        msg = self.connection.recv_match(

            type="GLOBAL_POSITION_INT",

            blocking=True

        )

        return {

            "lat": msg.lat / 1e7,

            "lon": msg.lon / 1e7,

            "alt": msg.relative_alt / 1000.0

        }

    # --------------------------------------------

    def read_attitude(self):

        msg = self.connection.recv_match(

            type="ATTITUDE",

            blocking=True

        )

        return {

            "roll": msg.roll,

            "pitch": msg.pitch,

            "yaw": msg.yaw

        }