import numpy as np
import matplotlib.pyplot as plt

import parameters as p
from motors import update_motor_speed

time = np.arange(
    0,
    1,
    p.dt
)

omega = np.zeros(4)

history = []

command = np.ones(4) * 500

for _ in time:

    omega = update_motor_speed(
        omega,
        command
    )

    history.append(omega[0])

plt.plot(time, history)

plt.grid(True)

plt.xlabel("Time (s)")

plt.ylabel("Motor Speed (rad/s)")

plt.title("Motor Step Response")

plt.show()