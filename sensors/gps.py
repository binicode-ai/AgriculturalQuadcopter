"""
gps.py

GPS sensor model.
"""

import numpy as np
import parameters as p

def gps_position(true_position):
    """
    Simulate GPS position measurement.
    """

    noise = np.random.normal(
        0.0,
        p.gps_position_std,
        3
    )

    return true_position + noise

def gps_velocity(true_velocity):
    """
    Simulate GPS velocity measurement.
    """

    noise = np.random.normal(
        0.0,
        p.gps_velocity_std,
        3
    )

    return true_velocity + noise

def should_update_gps(step):

    interval = int(
        1 /
        (p.gps_update_rate * p.dt)
    )

    return step % interval == 0