"""
magnetometer.py

3-axis magnetometer model.
"""

import numpy as np

import parameters as p
from rotation import rotation_matrix


def magnetometer(phi, theta, psi):
    """
    Simulate a 3-axis magnetometer.

    Parameters
    ----------
    phi, theta, psi : float
        Euler angles (rad)

    Returns
    -------
    ndarray (3,)
        Magnetic field measured in the body frame.
    """

    R = rotation_matrix(phi, theta, psi)

    magnetic_body = R.T @ p.mag_field

    noise = np.random.normal(
        0.0,
        p.mag_noise_std,
        3
    )

    return magnetic_body + noise