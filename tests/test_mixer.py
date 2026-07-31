import numpy as np

import parameters as p

from controllers.mixer import mix

hover_thrust = p.m * p.g

motor_speed = mix(
    hover_thrust,
    0.0,
    0.0,
    0.0
)

print("Motor Speeds")

print(motor_speed)