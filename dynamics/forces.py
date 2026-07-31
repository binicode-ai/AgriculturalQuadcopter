"""
forces.py

External forces acting on the quadcopter.
"""

import numpy as np
import parameters as p
import motors
from rotation import rotation_matrix

def thrust_force_world(phi,
                       theta,
                       psi,
                       omegas):
    """
    Convert thrust from body frame
    to world frame.
    """

    R = rotation_matrix(
        phi,
        theta,
        psi
    )

    return R @ thrust_force_body(
        omegas
    )

def thrust_force_body(omegas):
    """
    Total thrust in the body frame.
    """

    T = motors.total_thrust(omegas)

    return np.array([
        0.0,
        0.0,
        T
    ])

def gravity_force():
    """
    Gravity force in the inertial frame.

    Returns
    -------
    ndarray (3,)
    """

    return np.array([
        0.0,
        0.0,
        -p.m * p.g
    ])