"""
payload/trigger.py

Distance-based camera trigger.

Author: Biniyam Samuel
"""

import numpy as np


class DistanceTrigger:

    def __init__(self, capture_distance):

        self.capture_distance = capture_distance

        self.last_capture_position = None

    def should_capture(self, position):

        position = np.asarray(position, dtype=float)

        # First image
        if self.last_capture_position is None:

            self.last_capture_position = position.copy()

            return True

        distance = np.linalg.norm(
            position - self.last_capture_position
        )

        if distance >= self.capture_distance:

            self.last_capture_position = position.copy()

            return True

        return False

    def reset(self):

        self.last_capture_position = None