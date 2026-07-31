import numpy as np

import parameters as p
import motors

from rotation import rotation_matrix
from .aerodynamics import drag_force
# ==========================================================
# Rotational Dynamics
# ==========================================================


def rotational_acceleration(state, omegas):
    """
    Compute angular acceleration using the complete
    Newton-Euler rotational equation.

    I * omega_dot =
        tau - omega x (I * omega)
    """

    # Body angular velocity
    omega = np.array([
        state[9],   # p
        state[10],  # q
        state[11]   # r
    ])

    # Motor torques
    tau = np.array([
        motors.roll_torque(omegas),
        motors.pitch_torque(omegas),
        motors.yaw_torque(omegas)
    ])

    # Angular momentum
    H = p.I @ omega

    # Gyroscopic coupling
    gyro = np.cross(
        omega,
        H
    )

    # Complete Newton-Euler equation
    omega_dot = np.linalg.solve(
        p.I,
        tau - gyro
    )

    return omega_dot


    """
    Compute body angular acceleration.

    Current model ignores:

        ω × Iω

    This will be added later.
    """

    tau = np.array([
        motors.roll_torque(omegas),
        motors.pitch_torque(omegas),
        motors.yaw_torque(omegas)
    ])

    omega_dot = np.linalg.solve(
        p.I,
        tau
    )

    return omega_dot

"""
rotational.py

Rotational dynamics of the quadcopter.
"""


def angular_acceleration(omega, omegas):
    """
    Compute body angular acceleration.

    Parameters
    ----------
    omega : ndarray (3,)
        [p, q, r]

    omegas : ndarray (4,)
        Motor speeds.

    Returns
    -------
    ndarray (3,)
        Angular acceleration.
    """

    tau = np.array([
        motors.roll_torque(omegas),
        motors.pitch_torque(omegas),
        motors.yaw_torque(omegas)
    ])

    gyro = np.cross(
        omega,
        p.I @ omega
    )

    omega_dot = np.linalg.solve(
        p.I,
        tau - gyro
    )

    return omega_dot