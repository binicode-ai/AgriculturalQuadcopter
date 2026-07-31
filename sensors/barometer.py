"""
barometer.py

Simple barometric altitude sensor.
"""

import numpy as np

import parameters as p


def altitude(true_altitude):
    """
    Simulate a barometric altitude measurement.

    Parameters
    ----------
    true_altitude : float
        True altitude (m)

    Returns
    -------
    float
        Measured altitude.
    """

    noise = np.random.normal(
        0.0,
        p.baro_noise_std
    )

    return (
        true_altitude
        + p.baro_bias
        + noise
    )