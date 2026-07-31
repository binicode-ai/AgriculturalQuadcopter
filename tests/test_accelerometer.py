import numpy as np

from sensors.imu import accelerometer

phi = 0.0
theta = 0.0
psi = 0.0

linear_acc = np.zeros(3)

for i in range(10):

    measurement = accelerometer(
        phi,
        theta,
        psi,
        linear_acc
    )

    print(measurement)