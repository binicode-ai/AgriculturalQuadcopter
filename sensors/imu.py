"""
imu.py

IMU sensor model.

Includes:
- Gyroscope
- Accelerometer
"""

import numpy as np
import parameters as p

def gyroscope(true_rates):
    """
    Simulate a 3-axis gyroscope.

    Parameters
    ----------
    true_rates : ndarray (3,)
        True body rates [p, q, r].

    Returns
    -------
    ndarray (3,)
        Measured body rates.
    """

    noise = np.random.normal(
        0.0,
        p.gyro_noise_std,
        3
    )

    bias = np.array([
        p.gyro_bias_x,
        p.gyro_bias_y,
        p.gyro_bias_z
    ])

    return true_rates + bias + noise

import numpy as np
import parameters as p
from rotation import rotation_matrix


def accelerometer(phi,
                  theta,
                  psi,
                  linear_acc):
    """
    Simulate a 3-axis accelerometer.

    Parameters
    ----------
    phi, theta, psi : float
        Euler angles (rad)

    linear_acc : ndarray(3,)
        World-frame linear acceleration.

    Returns
    -------
    ndarray(3,)
        Accelerometer measurement in body frame.
    """

    R = rotation_matrix(phi, theta, psi)

    gravity = np.array([
        0.0,
        0.0,
        -p.g
    ])

    # Specific force in world frame
    specific_force_world = linear_acc - gravity

    # Convert to body frame
    specific_force_body = R.T @ specific_force_world

    bias = np.array([
        p.accel_bias_x,
        p.accel_bias_y,
        p.accel_bias_z
    ])

    noise = np.random.normal(
        0.0,
        p.accel_noise_std,
        3
    )

    return specific_force_body + bias + noise