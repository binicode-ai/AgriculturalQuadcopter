"""
Complementary Filter

Roll and Pitch estimation.
"""

import numpy as np

def estimate_roll(
    previous_roll,
    gyro_p,
    accel_y,
    accel_z,
    dt,
    alpha=0.98
):
    """
    Estimate roll angle.
    """

    # Gyroscope prediction
    gyro_roll = previous_roll + gyro_p * dt

    # Accelerometer estimate
    accel_roll = np.arctan2(
        accel_y,
        accel_z
    )

    # Complementary filter
    roll = (
        alpha * gyro_roll +
        (1 - alpha) * accel_roll
    )

    return roll

def estimate_pitch(
    previous_pitch,
    gyro_q,
    accel_x,
    accel_y,
    accel_z,
    dt,
    alpha=0.98
):
    """
    Estimate pitch angle.
    """

    gyro_pitch = previous_pitch + gyro_q * dt

    accel_pitch = np.arctan2(
        -accel_x,
        np.sqrt(accel_y**2 + accel_z**2)
    )

    pitch = (
        alpha * gyro_pitch +
        (1 - alpha) * accel_pitch
    )

    return pitch
