"""
rigid_body.py

State derivative for the quadcopter rigid-body dynamics.
"""

import numpy as np

from .translational import translational_acceleration
from .rotational import angular_acceleration
from kinematics import euler_rates


def state_derivative(state, omegas):
    """
    Compute the derivative of the 12-state vector.

    State Vector
    ------------
    [x, y, z,
     phi, theta, psi,
     vx, vy, vz,
     p, q, r]
    """

    (
        x,
        y,
        z,
        phi,
        theta,
        psi,
        vx,
        vy,
        vz,
        p_rate,
        q_rate,
        r_rate
    ) = state

    # ---------------------------------------
    # Translational Dynamics
    # ---------------------------------------

    linear_acc = translational_acceleration(
        phi,
        theta,
        psi,
        vx,
        vy,
        vz,
        omegas
    )

    # ---------------------------------------
    # Rotational Dynamics
    # ---------------------------------------

    omega = np.array([
        p_rate,
        q_rate,
        r_rate
    ])

    angular_acc = angular_acceleration(
        omega,
        omegas
    )

    # ---------------------------------------
    # Euler Angle Kinematics
    # ---------------------------------------

    phi_dot, theta_dot, psi_dot = euler_rates(
        phi,
        theta,
        p_rate,
        q_rate,
        r_rate
    )

    # ---------------------------------------
    # State Derivative
    # ---------------------------------------

    return np.array([

        # Position
        vx,
        vy,
        vz,

        # Euler Angles
        phi_dot,
        theta_dot,
        psi_dot,

        # Linear Velocity
        linear_acc[0],
        linear_acc[1],
        linear_acc[2],

        # Angular Velocity
        angular_acc[0],
        angular_acc[1],
        angular_acc[2]

    ])