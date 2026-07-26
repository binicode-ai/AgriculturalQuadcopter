"""
visualization.py

Plot quadcopter simulation results.

Author: Biniyam Samuel
"""

import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# Position
# ==========================================================

def plot_position(time, states):

    plt.figure(figsize=(10,5))

    plt.plot(time, states[:,0], label="x")
    plt.plot(time, states[:,1], label="y")
    plt.plot(time, states[:,2], label="z")

    plt.title("Position")

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()


# ==========================================================
# Velocity
# ==========================================================

def plot_velocity(time, states):

    plt.figure(figsize=(10,5))

    plt.plot(time, states[:,6], label="vx")
    plt.plot(time, states[:,7], label="vy")
    plt.plot(time, states[:,8], label="vz")

    plt.title("Linear Velocity")

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()


# ==========================================================
# Euler Angles
# ==========================================================

def plot_euler_angles(time, states):

    angles = np.degrees(states[:,3:6])

    plt.figure(figsize=(10,5))

    plt.plot(time, angles[:,0], label="Roll")

    plt.plot(time, angles[:,1], label="Pitch")

    plt.plot(time, angles[:,2], label="Yaw")

    plt.title("Euler Angles")

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()


# ==========================================================
# Angular Velocity
# ==========================================================

def plot_body_rates(time, states):

    rates = np.degrees(states[:,9:12])

    plt.figure(figsize=(10,5))

    plt.plot(time, rates[:,0], label="p")

    plt.plot(time, rates[:,1], label="q")

    plt.plot(time, rates[:,2], label="r")

    plt.title("Body Angular Rates")

    plt.xlabel("Time (s)")
    plt.ylabel("Angular Rate (deg/s)")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()


# ==========================================================
# Complete Dashboard
# ==========================================================

def show_all(time, states):

    plot_position(time, states)

    plot_velocity(time, states)

    plot_euler_angles(time, states)

    plot_body_rates(time, states)

    plt.show()