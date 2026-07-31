import matplotlib.pyplot as plt
from simulation import simulate
from visualization import show_all

import numpy as np
import parameters as p

state0 = np.zeros(12)

hover_speed = np.sqrt((p.m * p.g) / (4 * p.b))

omegas = np.array([
    hover_speed,
    1.05 * hover_speed,
    1.05 * hover_speed,
    hover_speed
])

"""
hover_speed = np.sqrt((p.m * p.g) / (4 * p.b))

climb_speed = 1.05 * hover_speed

omegas = np.full(4, climb_speed)
"""

time, states, estimated = simulate(
    state0,
    omegas,
    method="rk4"
)

print("Time shape:", time.shape)
print("States shape:", states.shape)

print("\nInitial State:")
print(states[0])

print("\nFinal State:")
print(states[-1])


show_all(time, states)