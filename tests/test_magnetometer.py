import numpy as np

from sensors.magnetometer import magnetometer

phi = 0.0
theta = 0.0

for yaw_deg in [0, 30, 60, 90, 180]:

    psi = np.radians(yaw_deg)

    measurement = magnetometer(
        phi,
        theta,
        psi
    )

    print(f"Yaw = {yaw_deg:3d}°  ->  {measurement}")