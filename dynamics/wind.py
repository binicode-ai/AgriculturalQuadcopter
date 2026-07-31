"""
wind.py

Wind disturbance models.
"""

import numpy as np
import parameters as p


def no_wind():
    """
    No wind.
    """
    return np.zeros(3)


def constant_wind():
    """
    Constant wind defined in parameters.py
    """
    return np.array([
        p.wind_x,
        p.wind_y,
        p.wind_z
    ])