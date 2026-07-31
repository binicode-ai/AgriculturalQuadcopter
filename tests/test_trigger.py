import numpy as np

from payload.trigger import DistanceTrigger

trigger = DistanceTrigger(
    capture_distance=5.0
)

for x in np.arange(0, 25, 1):

    position = np.array([x, 0, 10])

    if trigger.should_capture(position):

        print(
            f"Capture at x = {x}"
        )