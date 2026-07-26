"""
Yaw Test

Change clockwise/anticlockwise rotor speeds.

Expected:
Yaw rate changes.
"""

import numpy as np

import parameters as p
from simulation import simulate


def main():

    state0 = np.zeros(12)

    hover = np.sqrt((p.m * p.g) / (4 * p.b))

    omegas = np.array([
        1.02 * hover,
        hover,
        1.02 * hover,
        hover
    ])

    time, states = simulate(
    state0,
    omegas,
    method="rk4"
)

    print("=" * 60)
    print("YAW TEST")
    print("=" * 60)

    print("\nFinal Yaw Angle (rad)")
    print(states[-1, 5])

    print("\nFinal Yaw Rate (rad/s)")
    print(states[-1, 11])


if __name__ == "__main__":
    main()