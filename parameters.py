"""
parameters.py

Physical parameters of the Agricultural Quadcopter
"""

import numpy as np

# ===========================
# Physical Constants
# ===========================

# Gravity (m/s^2)
g = 9.81

# Mass of quadcopter (kg)
m = 1.5

# Distance from center to each motor (meters)
l = 0.25

# ===========================
# Inertia Matrix
# ===========================

Ix = 0.03
Iy = 0.03
Iz = 0.06

I = np.diag([Ix, Iy, Iz])

# ===========================
# Rotor Coefficients
# ===========================

# Lift coefficient
b = 3.13e-5

# Drag coefficient
d = 7.5e-7

# ===========================
# Simulation Parameters
# ===========================

dt = 0.01          # Time step (seconds)
simulation_time = 3.0