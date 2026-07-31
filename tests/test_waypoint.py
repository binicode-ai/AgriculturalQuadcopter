import numpy as np

from navigation.waypoint import WaypointNavigator

waypoints = [

    [0, 0, 5],

    [5, 0, 5],

    [5, 5, 5],

    [0, 5, 5]

]

navigator = WaypointNavigator(
    waypoints,
    tolerance=0.5
)

positions = [

    np.array([0.0, 0.0, 4.9]),

    np.array([5.1, 0.0, 5.0]),

    np.array([5.0, 5.1, 5.0]),

    np.array([0.1, 5.0, 5.0])

]

for i, pos in enumerate(positions):

    target = navigator.update(pos)

    print(f"Step {i + 1}")
    print("Current Position :", pos)
    print("Target Waypoint  :", target)
    print("Current Index    :", navigator.current_index)
    print()