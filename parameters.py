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

# =====================================================
# Aerodynamic Parameters
# =====================================================

drag_coefficient = 0.15      # N/(m/s)

air_density = 1.225          # kg/m^3

# =====================================================
# Aerodynamics
# =====================================================

rho = 1.225              # Air density (kg/m^3)

drag_x = 0.15            # N/(m/s)
drag_y = 0.15
drag_z = 0.20

# =====================================================
# Motor Dynamics
# =====================================================

motor_time_constant = 0.03      # seconds

motor_min_speed = 0.0

motor_max_speed = 900.0         # rad/s


# ==================================================
# Wind Parameters
# ==================================================

wind_x = 3.0      # m/s

wind_y = 0.0

wind_z = 0.0

# ==================================================
# IMU Parameters
# ==================================================

gyro_noise_std = 0.002      # rad/s

gyro_bias_x = 0.0
gyro_bias_y = 0.0
gyro_bias_z = 0.0

# ==================================================
# Accelerometer Parameters
# ==================================================

accel_noise_std = 0.05      # m/s²

accel_bias_x = 0.0
accel_bias_y = 0.0
accel_bias_z = 0.0

# ==================================================
# GPS Parameters
# ==================================================

gps_position_std = 1.5      # meters

gps_velocity_std = 0.15     # m/s

gps_update_rate = 10.0      # Hz

# ==================================================
# Magnetometer Parameters
# ==================================================

mag_noise_std = 0.01      # normalized units

# Earth's magnetic field (normalized)
mag_field = np.array([
    1.0,
    0.0,
    0.0
])

# ==================================================
# Barometer Parameters
# ==================================================

baro_noise_std = 0.10      # meters

baro_bias = 0.0            # meters