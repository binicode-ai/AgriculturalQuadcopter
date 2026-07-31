import numpy as np
import parameters as p

from controllers.flight_manager import FlightManager

manager = FlightManager()

state = np.zeros(12)

desired_position = np.array([5.0, 3.0, 10.0])

motor_speed = manager.update(

    desired_position,

    state,

    p.dt

)

print("Motor Speeds")

print(motor_speed)