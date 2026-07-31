import numpy as np

from sensors.gps import gps_position

true_position = np.array([
    100,
    50,
    20
])

for i in range(10):

    print(gps_position(true_position))