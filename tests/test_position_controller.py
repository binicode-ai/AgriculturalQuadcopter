import numpy as np
import parameters as p

from controllers.position import PositionController

controller = PositionController()

desired = np.array([10.0, 5.0, 20.0])

current = np.array([0.0, 0.0, 0.0])

for i in range(10):

    roll, pitch, thrust = controller.update(
        desired,
        current,
        p.dt
    )

    print(f"Step {i}")
    print(f"Roll   : {np.degrees(roll):.2f} deg")
    print(f"Pitch  : {np.degrees(pitch):.2f} deg")
    print(f"Thrust : {thrust:.2f} N")
    print()

    current += np.array([0.3, 0.15, 0.6])