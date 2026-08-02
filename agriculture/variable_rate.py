"""
agriculture/variable_rate.py

Variable-rate spraying controller.

Author: Biniyam Samuel
"""


class VariableRateController:

    def __init__(self):

        self.severity_multiplier = {

            "Low": 0.8,

            "Medium": 1.0,

            "High": 1.3,

            "None": 0.0,

            "Unknown": 1.0

        }

    # -------------------------------------------------

    def compute(

        self,

        base_duration,

        base_flow_rate,

        severity,

        confidence,

        altitude,

        speed

    ):

        multiplier = self.severity_multiplier.get(

            severity,

            1.0

        )

        # Reduce spraying if confidence is low
        confidence_factor = max(0.5, confidence)

        # Reduce flow at higher altitude
        altitude_factor = max(

            0.7,

            1.0 - 0.03 * max(0.0, altitude - 2.0)

        )

        # Increase flow slightly at higher speed
        speed_factor = min(

            1.3,

            1.0 + 0.05 * speed

        )

        duration = (

            base_duration
            * multiplier
            * confidence_factor
        )

        flow_rate = (

            base_flow_rate
            * altitude_factor
            * speed_factor
        )

        return {

            "duration": duration,

            "flow_rate": flow_rate,

            "multiplier": multiplier,

            "confidence_factor": confidence_factor,

            "altitude_factor": altitude_factor,

            "speed_factor": speed_factor

        }