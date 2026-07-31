import numpy as np

from sensors.imu import gyroscope

true_rate = np.array([
    0.5,
    0.0,
    -0.2
])

for i in range(10):

    measurement = gyroscope(true_rate)

    print(measurement)
    