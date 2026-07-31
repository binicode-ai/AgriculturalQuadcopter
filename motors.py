import numpy as np
import parameters as p


def rotor_thrust(omega):
    return p.b * omega**2


def total_thrust(omegas):
    thrusts = p.b * np.square(omegas)
    return np.sum(thrusts)


def roll_torque(omegas):
    w1, w2, w3, w4 = omegas
    return p.l * p.b * (w2**2 - w4**2)


def pitch_torque(omegas):
    w1, w2, w3, w4 = omegas
    return p.l * p.b * (w3**2 - w1**2)


def yaw_torque(omegas):
    w1, w2, w3, w4 = omegas
    return p.d * (
        w1**2
        - w2**2
        + w3**2
        - w4**2
    )


def control_inputs(omegas):
    U1 = total_thrust(omegas)
    U2 = roll_torque(omegas)
    U3 = pitch_torque(omegas)
    U4 = yaw_torque(omegas)

    return np.array([U1, U2, U3, U4])



def update_motor_speed(current_speed,
                       commanded_speed):
    """
    First-order motor dynamics.

    Parameters
    ----------
    current_speed : ndarray (4,)
    commanded_speed : ndarray (4,)

    Returns
    -------
    ndarray (4,)
    """

    omega_dot = (
        commanded_speed
        - current_speed
    ) / p.motor_time_constant

    current_speed = (
        current_speed
        + omega_dot * p.dt
    )

    current_speed = np.clip(
        current_speed,
        p.motor_min_speed,
        p.motor_max_speed
    )

    return current_speed

