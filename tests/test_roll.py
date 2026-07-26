"""
Roll Test

Increase right motors.

Expected:
Roll angle increases.
"""

import numpy as np

import parameters as p
from simulation import simulate


def main():

    state0 = np.zeros(12)

    hover = np.sqrt((p.m * p.g) / (4 * p.b))

    omegas = np.array([
        hover,
        1.05 * hover,
        1.05 * hover,
        hover
    ])

    time, states = simulate(
    state0,
    omegas,
    method="rk4"
)

    print("=" * 60)
    print("ROLL TEST")
    print("=" * 60)

    print("\nFinal Roll Angle (rad)")
    print(states[-1, 3])

    print("\nFinal Roll Rate (rad/s)")
    print(states[-1, 9])


if __name__ == "__main__":
    main()