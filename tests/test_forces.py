import numpy as np
import parameters as p
from dynamics.forces import gravity_force, thrust_force_body

hover_speed = np.sqrt((p.m * p.g) / (4 * p.b))
omegas = np.full(4, hover_speed)

Fg = gravity_force()
Ft = thrust_force_body(omegas)

print("Gravity Force:", Fg)
print("Thrust Force :", Ft)
print("Net Force    :", Fg + Ft)