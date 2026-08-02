"""
autopilot/failsafe.py

Central failsafe manager.

Author: Biniyam Samuel
"""


class FailsafeManager:

    def __init__(self):

        self.gps_ok = True

        self.imu_ok = True

        self.radio_ok = True

        self.battery_low = False

        self.battery_critical = False

    # --------------------------------------

    def update(

        self,

        gps_ok,

        imu_ok,

        radio_ok,

        battery_low,

        battery_critical

    ):

        self.gps_ok = gps_ok

        self.imu_ok = imu_ok

        self.radio_ok = radio_ok

        self.battery_low = battery_low

        self.battery_critical = battery_critical

    # --------------------------------------

    def decision(self):

        if self.battery_critical:

            return "LAND"

        if not self.imu_ok:

            return "LAND"

        if self.battery_low:

            return "RETURN_HOME"

        if not self.gps_ok:

            return "HOVER"

        if not self.radio_ok:

            return "RETURN_HOME"

        return "CONTINUE"

    # --------------------------------------

    def print_status(self):

        print()

        print("========== FAILSAFE ==========")

        print(f"GPS      : {self.gps_ok}")

        print(f"IMU      : {self.imu_ok}")

        print(f"Radio    : {self.radio_ok}")

        print(f"Low Batt : {self.battery_low}")

        print(f"Critical : {self.battery_critical}")

        print("------------------------------")

        print("Decision :", self.decision())

        print("==============================")