"""
Climb Test

Increase all motor speeds by 5%.

Expected:
Positive vertical acceleration.
"""

import numpy as np

import parameters as p
from simulation import simulate


def main():

    state0 = np.zeros(12)

    hover_speed = np.sqrt((p.m * p.g) / (4 * p.b))

    climb_speed = 1.05 * hover_speed

    omegas = np.full(4, climb_speed)

    time, states = simulate(
    state0,
    omegas,
    method="rk4"
)

    print("=" * 60)
    print("CLIMB TEST")
    print("=" * 60)

    print("\nMotor Speed")
    print(climb_speed)

    print("\nFinal Position")
    print(states[-1, 0:3])

    print("\nFinal Velocity")
    print(states[-1, 6:9])


if __name__ == "__main__":
    main()