"""
Pitch Test

Increase rear motors.

Expected:
Pitch angle changes.
"""

import numpy as np

import parameters as p
from simulation import simulate


def main():

    state0 = np.zeros(12)

    hover = np.sqrt((p.m * p.g) / (4 * p.b))

    omegas = np.array([
        hover,
        hover,
        1.05 * hover,
        1.05 * hover
    ])

    time, states = simulate(
    state0,
    omegas,
    method="rk4"
)

    print("=" * 60)
    print("PITCH TEST")
    print("=" * 60)

    print("\nFinal Pitch Angle (rad)")
    print(states[-1, 4])

    print("\nFinal Pitch Rate (rad/s)")
    print(states[-1, 10])


if __name__ == "__main__":
    main()