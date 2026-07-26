"""
Hover Test

Expected Result:
- Position remains approximately constant.
- Velocity remains close to zero.
"""

import numpy as np

import parameters as p
from simulation import simulate


def main():

    state0 = np.zeros(12)

    hover_speed = np.sqrt((p.m * p.g) / (4 * p.b))

    omegas = np.array([
        hover_speed,
        hover_speed,
        hover_speed,
        hover_speed
    ])

    time, states = simulate(
    state0,
    omegas,
    method="rk4"
)

    print("=" * 60)
    print("HOVER TEST")
    print("=" * 60)

    print("\nHover Speed")
    print(f"{hover_speed:.2f} rad/s")

    print("\nFinal Position [x y z]")
    print(states[-1, 0:3])

    print("\nFinal Euler Angles [roll pitch yaw]")
    print(states[-1, 3:6])

    print("\nFinal Linear Velocity")
    print(states[-1, 6:9])

    print("\nFinal Angular Velocity")
    print(states[-1, 9:12])


if __name__ == "__main__":
    main()