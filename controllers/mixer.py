"""
controllers/mixer.py

Motor mixer for X-frame quadcopter.

Converts:

    Total thrust
    Roll torque
    Pitch torque
    Yaw torque

into

    Four motor angular speeds.

Author: Biniyam Samuel
"""

import numpy as np
import parameters as p


def mix(total_thrust, tau_roll, tau_pitch, tau_yaw):
    """
    Convert desired thrust and body torques
    into motor angular speeds.

    Parameters
    ----------
    total_thrust : float
        Total thrust (N)

    tau_roll : float
        Desired roll torque (N·m)

    tau_pitch : float
        Desired pitch torque (N·m)

    tau_yaw : float
        Desired yaw torque (N·m)

    Returns
    -------
    omega : ndarray (4,)
        Motor angular speeds (rad/s)
    """

    # Arm length
    L = p.l

    # Drag coefficient
    d = p.d

    # -------------------------------------------------
    # Motor thrust allocation (X configuration)
    # -------------------------------------------------

    F1 = (
        total_thrust / 4
        - tau_pitch / (2 * L)
        + tau_yaw / (4 * d)
    )

    F2 = (
        total_thrust / 4
        + tau_roll / (2 * L)
        - tau_yaw / (4 * d)
    )

    F3 = (
        total_thrust / 4
        + tau_pitch / (2 * L)
        + tau_yaw / (4 * d)
    )

    F4 = (
        total_thrust / 4
        - tau_roll / (2 * L)
        - tau_yaw / (4 * d)
    )

    thrusts = np.array([
        F1,
        F2,
        F3,
        F4
    ])

    # -------------------------------------------------
    # Prevent negative thrust
    # -------------------------------------------------

    thrusts = np.maximum(thrusts, 0.0)

    # -------------------------------------------------
    # Limit maximum thrust per motor
    #
    # A conservative limit:
    # each motor should not exceed
    # twice the hover thrust.
    # -------------------------------------------------

    hover_per_motor = (p.m * p.g) / 4

    max_thrust = 2.0 * hover_per_motor

    thrusts = np.clip(
        thrusts,
        0.0,
        max_thrust
    )

    # -------------------------------------------------
    # Convert thrust to motor speed
    #
    # F = b * omega²
    # -------------------------------------------------

    omega = np.sqrt(thrusts / p.b)

    # -------------------------------------------------
    # Motor speed saturation
    # -------------------------------------------------

    omega = np.clip(
        omega,
        p.motor_min_speed,
        p.motor_max_speed
    )

    # -------------------------------------------------
    # Debug Information
    # -------------------------------------------------

    print("\n========== MOTOR MIXER ==========")
    print(f"Total Thrust : {total_thrust:.3f} N")
    print(f"Roll Torque  : {tau_roll:.4f} N·m")
    print(f"Pitch Torque : {tau_pitch:.4f} N·m")
    print(f"Yaw Torque   : {tau_yaw:.4f} N·m")
    print("---------------------------------")
    print("Motor Thrusts (N)")
    print(thrusts)
    print("---------------------------------")
    print("Motor Speeds (rad/s)")
    print(omega)
    print("=================================\n")

    return omega