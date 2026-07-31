"""
simulation.py

Runs the quadcopter simulation.

Supports:
    - Euler Integration
    - Runge-Kutta 4 (RK4) Integration
"""

import numpy as np

import parameters as p
from dynamics import rk4_step
from dynamics import euler_step

def simulate(initial_state, omegas, method="rk4"):
    """
    Simulate the quadcopter.

    Parameters
    ----------
    initial_state : ndarray
        Initial 12-state vector.

    omegas : ndarray
        Four motor angular speeds (rad/s).

    method : str
        Integration method:
            "euler" -> Forward Euler
            "rk4"   -> Runge-Kutta 4

    Returns
    -------
    time : ndarray
        Simulation time vector.

    states : ndarray
        History of the state vector.
    """

    # Number of simulation steps
    steps = int(p.simulation_time / p.dt)

    # Time vector
    time = np.linspace(
        0.0,
        p.simulation_time,
        steps + 1
    )

    # Allocate memory
    states = np.zeros((steps + 1, 12))

    # Initial condition
    states[0] = initial_state

    # Simulation loop
    for k in range(steps):

        if method.lower() == "euler":

            states[k + 1] = euler_step(
                states[k],
                omegas
            )

        elif method.lower() == "rk4":

            states[k + 1] = rk4_step(
                states[k],
                omegas
            )

        else:
            raise ValueError(
                f"Unknown integration method: {method}\n"
                "Choose 'euler' or 'rk4'."
            )

    return time, states

