import numpy as np

import parameters as p
import motors

from rotation import rotation_matrix
from .aerodynamics import drag_force


def translational_acceleration(
    phi,
    theta,
    psi,
    vx,
    vy,
    vz,
    omegas
):

    thrust = motors.total_thrust(omegas)

    thrust_body = np.array([
        0.0,
        0.0,
        thrust
    ])

    R = rotation_matrix(phi, theta, psi)

    thrust_world = R @ thrust_body

    gravity = np.array([
        0.0,
        0.0,
        -p.g
    ])

    velocity = np.array([
        vx,
        vy,
        vz
    ])

    drag = drag_force(velocity)

    force = (
        thrust_world
        + drag
        + p.m * gravity
    )

    return force / p.m