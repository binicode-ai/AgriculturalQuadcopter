"""
payload/logger.py

Flight logger for agricultural survey missions.

Author: Biniyam Samuel
"""

import csv


class FlightLogger:

    def __init__(self):

        self.records = []

    def log(

        self,

        image_id,

        timestamp,

        position,

        attitude,

        velocity,

        waypoint

    ):

        self.records.append({

            "image_id": image_id,

            "time": timestamp,

            "x": position[0],

            "y": position[1],

            "z": position[2],

            "roll": attitude[0],

            "pitch": attitude[1],

            "yaw": attitude[2],

            "vx": velocity[0],

            "vy": velocity[1],

            "vz": velocity[2],

            "waypoint": waypoint

        })

    def save_csv(

        self,

        filename="flight_log.csv"

    ):

        if len(self.records) == 0:

            print("No flight records.")

            return

        keys = self.records[0].keys()

        with open(

            filename,

            "w",

            newline=""

        ) as file:

            writer = csv.DictWriter(

                file,

                fieldnames=keys

            )

            writer.writeheader()

            writer.writerows(

                self.records

            )

        print(

            f"Flight log saved to {filename}"

        )