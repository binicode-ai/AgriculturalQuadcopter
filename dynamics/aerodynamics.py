"""
aerodynamics.py

Aerodynamic drag model.
"""

import numpy as np
import parameters as p


def drag_force(relative_velocity):
    """
    Compute linear aerodynamic drag.

    Parameters
    ----------
    relative_velocity : ndarray (3,)
        Relative air velocity [vx, vy, vz]

    Returns
    -------
    ndarray (3,)
        Drag force [Fx, Fy, Fz]
    """

    drag = np.array([
        -p.drag_x * relative_velocity[0],
        -p.drag_y * relative_velocity[1],
        -p.drag_z * relative_velocity[2]
    ])

    return drag