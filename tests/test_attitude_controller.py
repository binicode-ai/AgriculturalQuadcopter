import numpy as np

import parameters as p

from controllers.attitude import AttitudeController

controller = AttitudeController()

desired_roll = np.radians(10)

desired_pitch = 0.0

desired_yaw = 0.0

estimated_roll = 0.0

estimated_pitch = 0.0

estimated_yaw = 0.0

for i in range(50):

    tau = controller.update(

        desired_roll,

        desired_pitch,

        desired_yaw,

        estimated_roll,

        estimated_pitch,

        estimated_yaw,

        p.dt

    )

    estimated_roll += tau[0] * 0.002

    print(

        f"Step {i:2d}",

        f"Roll = {np.degrees(estimated_roll):6.2f}°",

        f"Torque = {tau[0]:7.3f}"

    )