import numpy as np

from estimation.complementary import (
    estimate_roll,
    estimate_pitch
)

roll = 0.0
pitch = 0.0

dt = 0.01

gyro_p = 0.1
gyro_q = 0.05

accel = np.array([
    0.0,
    0.0,
    9.81
])

for i in range(100):

    roll = estimate_roll(
        roll,
        gyro_p,
        accel[1],
        accel[2],
        dt
    )

    pitch = estimate_pitch(
        pitch,
        gyro_q,
        accel[0],
        accel[1],
        accel[2],
        dt
    )

print(f"Estimated Roll:  {np.degrees(roll):.2f}°")
print(f"Estimated Pitch: {np.degrees(pitch):.2f}°")