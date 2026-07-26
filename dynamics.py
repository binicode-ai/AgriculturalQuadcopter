"""
dynamics.py

Newton-Euler dynamics of the quadcopter.

Current Version
---------------
✔ Translational dynamics
✔ Rotational dynamics
✔ Correct Euler kinematics
✔ Euler numerical integration

Future Versions
---------------
- Gyroscopic coupling
- Aerodynamic drag
- RK4 integration
"""

import numpy as np

import parameters as p
import motors

from rotation import rotation_matrix
from kinematics import euler_rates


# ==========================================================
# Translational Dynamics
# ==========================================================

def translational_acceleration(phi, theta, psi, omegas):
    """
    Compute linear acceleration in the Earth frame.
    """

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

    acceleration = thrust_world / p.m + gravity

    return acceleration


# ==========================================================
# Rotational Dynamics
# ==========================================================


def rotational_acceleration(state, omegas):
    """
    Compute angular acceleration using the complete
    Newton-Euler rotational equation.

    I * omega_dot =
        tau - omega x (I * omega)
    """

    # Body angular velocity
    omega = np.array([
        state[9],   # p
        state[10],  # q
        state[11]   # r
    ])

    # Motor torques
    tau = np.array([
        motors.roll_torque(omegas),
        motors.pitch_torque(omegas),
        motors.yaw_torque(omegas)
    ])

    # Angular momentum
    H = p.I @ omega

    # Gyroscopic coupling
    gyro = np.cross(
        omega,
        H
    )

    # Complete Newton-Euler equation
    omega_dot = np.linalg.solve(
        p.I,
        tau - gyro
    )

    return omega_dot


    """
    Compute body angular acceleration.

    Current model ignores:

        ω × Iω

    This will be added later.
    """

    tau = np.array([
        motors.roll_torque(omegas),
        motors.pitch_torque(omegas),
        motors.yaw_torque(omegas)
    ])

    omega_dot = np.linalg.solve(
        p.I,
        tau
    )

    return omega_dot


# ==========================================================
# State Derivatives
# ==========================================================

def state_derivative(state, omegas):
    """
    Compute derivative of the 12-state vector.

    State

    x
    y
    z

    phi
    theta
    psi

    vx
    vy
    vz

    p
    q
    r
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
    # Linear acceleration
    # ---------------------------------------

    linear_acc = translational_acceleration(
        phi,
        theta,
        psi,
        omegas
    )

    # ---------------------------------------
    # Angular acceleration
    # ---------------------------------------

    angular_acc = rotational_acceleration(
    state,
    omegas
)

    # ---------------------------------------
    # Euler angle derivatives
    # ---------------------------------------

    phi_dot, theta_dot, psi_dot = euler_rates(
        phi,
        theta,
        p_rate,
        q_rate,
        r_rate
    )

    # ---------------------------------------
    # Return derivative
    # ---------------------------------------

    return np.array([

        # Position

        vx,
        vy,
        vz,

        # Orientation

        phi_dot,
        theta_dot,
        psi_dot,

        # Velocity

        linear_acc[0],
        linear_acc[1],
        linear_acc[2],

        # Angular Velocity

        angular_acc[0],
        angular_acc[1],
        angular_acc[2]

    ])


# ==========================================================
# Euler Integrator
# ==========================================================

def euler_step(state, omegas):
    """
    Euler Integration

    x(k+1)=x(k)+f(x,u)dt
    """

    derivative = state_derivative(
        state,
        omegas
    )

    next_state = state + derivative * p.dt

    return next_state

# ==========================================================
# Runge-Kutta 4 Integrator
# ==========================================================

def rk4_step(state, omegas):
    """
    Fourth-order Runge-Kutta integration.

    Parameters
    ----------
    state : ndarray
        Current state vector.

    omegas : ndarray
        Motor angular speeds.

    Returns
    -------
    ndarray
        State at the next time step.
    """

    dt = p.dt

    k1 = state_derivative(state, omegas)

    k2 = state_derivative(
        state + 0.5 * dt * k1,
        omegas
    )

    k3 = state_derivative(
        state + 0.5 * dt * k2,
        omegas
    )

    k4 = state_derivative(
        state + dt * k3,
        omegas
    )

    next_state = state + (dt / 6.0) * (
        k1 +
        2 * k2 +
        2 * k3 +
        k4
    )

    return next_state