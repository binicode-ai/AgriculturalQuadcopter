
"""
main.py

AgriculturalQuadcopter
Complete Quadcopter Simulation Dashboard

Uses:
    simulation.simulate()

Simulation state vector:
    0  = X position
    1  = Y position
    2  = Z position

    3  = X velocity
    4  = Y velocity
    5  = Z velocity

    6  = Roll
    7  = Pitch
    8  = Yaw

    9  = Roll rate
    10 = Pitch rate
    11 = Yaw rate

Outputs:
    1. X position
    2. Y position
    3. Z position / altitude
    4. X velocity
    5. Y velocity
    6. Z velocity
    7. Roll
    8. Pitch
    9. Yaw
    10. Roll rate
    11. Pitch rate
    12. Yaw rate
    13. 3D flight trajectory

The simulation itself remains in simulation.py.
This file is only responsible for running and visualizing it.
"""

import os
import traceback

import numpy as np
import matplotlib.pyplot as plt

from simulation import simulate
import parameters as p


# ============================================================
# Configuration
# ============================================================

SIMULATION_METHOD = "rk4"

# Initial 12-state vector
#
# [x, y, z,
#  vx, vy, vz,
#  roll, pitch, yaw,
#  p, q, r]
#
# Units:
# position       -> meters
# velocity       -> m/s
# attitude       -> radians
# angular rate   -> rad/s

INITIAL_STATE = np.zeros(12, dtype=float)


# ------------------------------------------------------------
# Initial altitude
# ------------------------------------------------------------
#
# Change this if your coordinate system requires another
# starting altitude.
#
# For a ground-level simulation:
INITIAL_STATE[2] = 0.0


# ------------------------------------------------------------
# Initial attitude
# ------------------------------------------------------------

INITIAL_STATE[6] = 0.0       # roll
INITIAL_STATE[7] = 0.0       # pitch
INITIAL_STATE[8] = 0.0       # yaw


# ------------------------------------------------------------
# Motor angular velocities
# ------------------------------------------------------------
#
# These values are intentionally kept together so that you can
# easily modify them for your quadcopter model.
#
# Units: rad/s
#
# If your parameters.py already contains motor speeds, replace
# these values with those parameters.

OMEGAS = np.array([
    200.0,
    200.0,
    400.0,
    400.0,
], dtype=float)


# ============================================================
# Helper functions
# ============================================================

def radians_to_degrees(values):
    """Convert radians to degrees."""
    return np.rad2deg(values)


def print_simulation_information(
    time,
    states,
):
    """Print a concise simulation summary."""

    print()
    print("=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print()

    print(
        f"Simulation method : {SIMULATION_METHOD.upper()}"
    )

    print(
        f"Simulation time   : {time[-1]:.3f} s"
    )

    print(
        f"Time steps        : {len(time) - 1}"
    )

    print(
        f"Time step         : {p.dt:.6f} s"
    )

    print()

    print("FINAL STATE")
    print("-" * 60)

    print(
        f"X position        : {states[-1, 0]: .6f} m"
    )

    print(
        f"Y position        : {states[-1, 1]: .6f} m"
    )

    print(
        f"Z position        : {states[-1, 2]: .6f} m"
    )

    print()

    print(
        f"X velocity        : {states[-1, 3]: .6f} m/s"
    )

    print(
        f"Y velocity        : {states[-1, 4]: .6f} m/s"
    )

    print(
        f"Z velocity        : {states[-1, 5]: .6f} m/s"
    )

    print()

    print(
        f"Roll              : "
        f"{np.rad2deg(states[-1, 6]): .6f} deg"
    )

    print(
        f"Pitch             : "
        f"{np.rad2deg(states[-1, 7]): .6f} deg"
    )

    print(
        f"Yaw               : "
        f"{np.rad2deg(states[-1, 8]): .6f} deg"
    )

    print()

    print(
        f"Roll rate         : "
        f"{states[-1, 9]: .6f} rad/s"
    )

    print(
        f"Pitch rate        : "
        f"{states[-1, 10]: .6f} rad/s"
    )

    print(
        f"Yaw rate          : "
        f"{states[-1, 11]: .6f} rad/s"
    )

    print()


# ============================================================
# Plot 1
# Position
# ============================================================

def plot_position(time, states):

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    axes[0].plot(
        time,
        states[:, 0],
    )

    axes[0].set_ylabel("X (m)")
    axes[0].set_title("X Position")

    axes[1].plot(
        time,
        states[:, 1],
    )

    axes[1].set_ylabel("Y (m)")
    axes[1].set_title("Y Position")

    axes[2].plot(
        time,
        states[:, 2],
    )

    axes[2].set_ylabel("Z (m)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Z Position / Altitude")

    fig.suptitle(
        "Qu adcopter Position",
        fontsize=14,
    )

    fig.tight_layout()

    return fig


# ============================================================
# Plot 2
# Velocity
# ============================================================

def plot_velocity(time, states):

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    axes[0].plot(
        time,
        states[:, 3],
    )

    axes[0].set_ylabel("m/s")
    axes[0].set_title("X Velocity")

    axes[1].plot(
        time,
        states[:, 4],
    )

    axes[1].set_ylabel("m/s")
    axes[1].set_title("Y Velocity")

    axes[2].plot(
        time,
        states[:, 5],
    )

    axes[2].set_ylabel("m/s")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Z Velocity")

    fig.suptitle(
        "Qu adcopter Linear Velocity",
        fontsize=14,
    )

    fig.tight_layout()

    return fig


# ============================================================
# Plot 3
# Attitude
# ============================================================

def plot_attitude(time, states):

    roll = radians_to_degrees(
        states[:, 6]
    )

    pitch = radians_to_degrees(
        states[:, 7]
    )

    yaw = radians_to_degrees(
        states[:, 8]
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    axes[0].plot(
        time,
        roll,
    )

    axes[0].set_ylabel("deg")
    axes[0].set_title("Roll")

    axes[1].plot(
        time,
        pitch,
    )

    axes[1].set_ylabel("deg")
    axes[1].set_title("Pitch")

    axes[2].plot(
        time,
        yaw,
    )

    axes[2].set_ylabel("deg")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Yaw")

    fig.suptitle(
        "Qu adcopter Attitude",
        fontsize=14,
    )

    fig.tight_layout()

    return fig


# ============================================================
# Plot 4
# Angular velocity
# ============================================================

def plot_angular_velocity(time, states):

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    axes[0].plot(
        time,
        states[:, 9],
    )

    axes[0].set_ylabel("rad/s")
    axes[0].set_title("Roll Rate (p)")

    axes[1].plot(
        time,
        states[:, 10],
    )

    axes[1].set_ylabel("rad/s")
    axes[1].set_title("Pitch Rate (q)")

    axes[2].plot(
        time,
        states[:, 11],
    )

    axes[2].set_ylabel("rad/s")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Yaw Rate (r)")

    fig.suptitle(
        "Qu adcopter Angular Velocity",
        fontsize=14,
    )

    fig.tight_layout()

    return fig


# ============================================================
# Plot 5
# Combined position
# ============================================================

def plot_combined_position(time, states):

    fig = plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        time,
        states[:, 0],
        label="X",
    )

    plt.plot(
        time,
        states[:, 1],
        label="Y",
    )

    plt.plot(
        time,
        states[:, 2],
        label="Z",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title(
        "Position Components"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    return fig


# ============================================================
# Plot 6
# Combined velocity
# ============================================================

def plot_combined_velocity(time, states):

    fig = plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        time,
        states[:, 3],
        label="Vx",
    )

    plt.plot(
        time,
        states[:, 4],
        label="Vy",
    )

    plt.plot(
        time,
        states[:, 5],
        label="Vz",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title(
        "Velocity Components"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    return fig


# ============================================================
# Plot 7
# Combined attitude
# ============================================================

def plot_combined_attitude(time, states):

    fig = plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        time,
        np.rad2deg(states[:, 6]),
        label="Roll",
    )

    plt.plot(
        time,
        np.rad2deg(states[:, 7]),
        label="Pitch",
    )

    plt.plot(
        time,
        np.rad2deg(states[:, 8]),
        label="Yaw",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title(
        "Attitude Components"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    return fig


# ============================================================
# Plot 8
# Combined angular velocity
# ============================================================

def plot_combined_angular_velocity(
    time,
    states,
):

    fig = plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        time,
        states[:, 9],
        label="p - Roll rate",
    )

    plt.plot(
        time,
        states[:, 10],
        label="q - Pitch rate",
    )

    plt.plot(
        time,
        states[:, 11],
        label="r - Yaw rate",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Angular velocity (rad/s)")
    plt.title(
        "Angular Velocity Components"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    return fig


# ============================================================
# Plot 9
# 3D trajectory
# ============================================================

def plot_3d_trajectory(states):

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]

    ax.plot(
        x,
        y,
        z,
    )

    ax.scatter(
        x[0],
        y[0],
        z[0],
        s=60,
        label="Start",
    )

    ax.scatter(
        x[-1],
        y[-1],
        z[-1],
        s=60,
        label="End",
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.set_title(
        "3D Quadcopter Flight Trajectory"
    )

    ax.legend()

    plt.tight_layout()

    return fig


# ============================================================
# Plot 10
# XY ground trajectory
# ============================================================

def plot_xy_trajectory(states):

    fig = plt.figure(
        figsize=(9, 7)
    )

    plt.plot(
        states[:, 0],
        states[:, 1],
    )

    plt.scatter(
        states[0, 0],
        states[0, 1],
        s=70,
        label="Start",
    )

    plt.scatter(
        states[-1, 0],
        states[-1, 1],
        s=70,
        label="End",
    )

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.title(
        "Horizontal Flight Trajectory"
    )

    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    return fig


# ============================================================
# Plot 11
# Altitude
# ============================================================

def plot_altitude(time, states):

    fig = plt.figure(
        figsize=(10, 6)
    )

    altitude = states[:, 2]

    plt.plot(
        time,
        altitude,
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")

    plt.title(
        "Quadcopter Altitude Profile"
    )

    plt.grid(True)

    plt.tight_layout()

    return fig


# ============================================================
# Plot 12
# Full state dashboard
# ============================================================

def plot_full_state_dashboard(
    time,
    states,
):

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(15, 13),
        sharex=True,
    )

    # Position
    axes[0, 0].plot(
        time,
        states[:, 0],
    )

    axes[0, 0].set_title(
        "X Position"
    )

    axes[0, 0].set_ylabel(
        "m"
    )

    axes[0, 1].plot(
        time,
        states[:, 1],
    )

    axes[0, 1].set_title(
        "Y Position"
    )

    axes[0, 2].plot(
        time,
        states[:, 2],
    )

    axes[0, 2].set_title(
        "Z Position"
    )

    # Velocity
    axes[1, 0].plot(
        time,
        states[:, 3],
    )

    axes[1, 0].set_title(
        "X Velocity"
    )

    axes[1, 0].set_ylabel(
        "m/s"
    )

    axes[1, 1].plot(
        time,
        states[:, 4],
    )

    axes[1, 1].set_title(
        "Y Velocity"
    )

    axes[1, 2].plot(
        time,
        states[:, 5],
    )

    axes[1, 2].set_title(
        "Z Velocity"
    )

    # Attitude
    axes[2, 0].plot(
        time,
        np.rad2deg(states[:, 6]),
    )

    axes[2, 0].set_title(
        "Roll"
    )

    axes[2, 0].set_ylabel(
        "deg"
    )

    axes[2, 1].plot(
        time,
        np.rad2deg(states[:, 7]),
    )

    axes[2, 1].set_title(
        "Pitch"
    )

    axes[2, 2].plot(
        time,
        np.rad2deg(states[:, 8]),
    )

    axes[2, 2].set_title(
        "Yaw"
    )

    # Angular velocity
    axes[3, 0].plot(
        time,
        states[:, 9],
    )

    axes[3, 0].set_title(
        "Roll Rate"
    )

    axes[3, 0].set_ylabel(
        "rad/s"
    )

    axes[3, 1].plot(
        time,
        states[:, 10],
    )

    axes[3, 1].set_title(
        "Pitch Rate"
    )

    axes[3, 2].plot(
        time,
        states[:, 11],
    )

    axes[3, 2].set_title(
        "Yaw Rate"
    )

    for row in axes:

        for axis in row:

            axis.grid(True)

            axis.set_xlabel(
                "Time (s)"
            )

    fig.suptitle(
        "AgriculturalQuadcopter - Complete 12-State Simulation",
        fontsize=16,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    return fig


# ============================================================
# Run simulation
# ============================================================

def run_simulation():

    print()
    print("=" * 60)
    print("AgriculturalQuadcopter")
    print("MAIN SIMULATION")
    print("=" * 60)
    print()

    print(
        "Running quadcopter simulation..."
    )

    print()

    print(
        f"Integration method: "
        f"{SIMULATION_METHOD.upper()}"
    )

    print(
        f"Simulation time: "
        f"{p.simulation_time:.3f} s"
    )

    print(
        f"Time step: "
        f"{p.dt:.6f} s"
    )

    print()

    print(
        "Initial state:"
    )

    print(
        INITIAL_STATE
    )

    print()

    print(
        "Motor speeds (rad/s):"
    )

    print(
        OMEGAS
    )

    print()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # simulation.py returns:
    #
    #     time, states
    #
    # NOT:
    #
    #     time, states, estimated
    #
    # --------------------------------------------------------

    time, states = simulate(
        INITIAL_STATE,
        OMEGAS,
        method=SIMULATION_METHOD,
    )

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    if len(time) != len(states):

        raise RuntimeError(
            "Simulation output mismatch:\n"
            f"time length = {len(time)}\n"
            f"states length = {len(states)}"
        )

    if states.ndim != 2:

        raise RuntimeError(
            "Expected states to be a 2D array."
        )

    if states.shape[1] != 12:

        raise RuntimeError(
            "Expected 12 state variables, "
            f"but received {states.shape[1]}."
        )

    print(
        "Simulation completed successfully."
    )

    print()

    print(
        f"State array shape: "
        f"{states.shape}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_simulation_information(
        time,
        states,
    )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    print(
        "Generating complete simulation plots..."
    )

    print()

    figures = []

    print(
        "1/13  Position plots..."
    )

    figures.append(
        plot_position(
            time,
            states,
        )
    )

    print(
        "2/13  Velocity plots..."
    )

    figures.append(
        plot_velocity(
            time,
            states,
        )
    )

    print(
        "3/13  Attitude plots..."
    )

    figures.append(
        plot_attitude(
            time,
            states,
        )
    )

    print(
        "4/13  Angular velocity plots..."
    )

    figures.append(
        plot_angular_velocity(
            time,
            states,
        )
    )

    print(
        "5/13  Combined position..."
    )

    figures.append(
        plot_combined_position(
            time,
            states,
        )
    )

    print(
        "6/13  Combined velocity..."
    )

    figures.append(
        plot_combined_velocity(
            time,
            states,
        )
    )

    print(
        "7/13  Combined attitude..."
    )

    figures.append(
        plot_combined_attitude(
            time,
            states,
        )
    )

    print(
        "8/13  Combined angular velocity..."
    )

    figures.append(
        plot_combined_angular_velocity(
            time,
            states,
        )
    )

    print(
        "9/13  3D trajectory..."
    )

    figures.append(
        plot_3d_trajectory(
            states,
        )
    )

    print(
        "10/13 XY trajectory..."
    )

    figures.append(
        plot_xy_trajectory(
            states,
        )
    )

    print(
        "11/13 Altitude..."
    )

    figures.append(
        plot_altitude(
            time,
            states,
        )
    )

    print(
        "12/13 Full state dashboard..."
    )

    figures.append(
        plot_full_state_dashboard(
            time,
            states,
        )
    )

    # --------------------------------------------------------
    # Final display
    # --------------------------------------------------------

    print()
    print(
        "All simulation plots generated."
    )

    print(
        "Close the plot windows to finish."
    )

    print()

    plt.show()

    return time, states


# ============================================================
# Main
# ============================================================

def main():

    try:

        run_simulation()

    except Exception as exc:

        print()
        print("=" * 60)
        print("SIMULATION FAILED")
        print("=" * 60)
        print()

        print(
            f"Error: {exc}"
        )

        print()

        traceback.print_exc()

        print()

        raise


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()

