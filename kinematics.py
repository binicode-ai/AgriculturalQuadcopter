"""
kinematics.py

Conversion between body angular rates and Euler angle rates.
"""

import numpy as np


def euler_rate_matrix(phi, theta):
    """
    Transformation matrix from body rates [p, q, r]
    to Euler angle rates [phi_dot, theta_dot, psi_dot].
    """

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    tan_theta = np.tan(theta)
    cos_theta = np.cos(theta)

    # Prevent division by zero
    if abs(cos_theta) < 1e-6:
        raise ValueError(
            "Pitch angle is too close to ±90°. Euler angles become singular."
        )

    return np.array([
        [1, sin_phi * tan_theta, cos_phi * tan_theta],
        [0, cos_phi,            -sin_phi],
        [0, sin_phi / cos_theta, cos_phi / cos_theta]
    ])


def euler_rates(phi, theta, p_rate, q_rate, r_rate):
    """
    Compute Euler angle derivatives.
    """

    T = euler_rate_matrix(phi, theta)

    body_rates = np.array([
        p_rate,
        q_rate,
        r_rate
    ])

    return T @ body_rates