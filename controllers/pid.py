"""
controller/pid.py

Generic PID Controller
"""

class PID:

    def __init__(
        self,
        kp,
        ki,
        kd,
        integral_limit=5.0
    ):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0
        self.previous_error = 0.0

        self.integral_limit = integral_limit

    def update(
        self,
        target,
        measurement,
        dt
    ):

        # Error
        error = target - measurement

        # Integral term
        self.integral += error * dt

        # Anti-windup
        self.integral = max(
            -self.integral_limit,
            min(self.integral_limit, self.integral)
        )

        # Derivative term
        derivative = (
            error - self.previous_error
        ) / dt

        self.previous_error = error

        # PID output
        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        return output

    def reset(self):
        """
        Reset the PID controller state.
        """
        self.integral = 0.0
        self.previous_error = 0.0