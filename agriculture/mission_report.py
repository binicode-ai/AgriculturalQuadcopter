"""
agriculture/mission_report.py

Mission report generator.

Author: Biniyam Samuel
"""

from collections import Counter
from datetime import datetime


class MissionReport:

    def __init__(self):

        self.start_time = datetime.now()

        self.end_time = None

        self.distance = 0.0

        self.inspected = 0

        self.sprayed = 0

        self.chemical_used = 0.0

        self.remaining_tank = 0.0

        self.diseases = []

    # ---------------------------------------

    def add_detection(self, disease):

        self.inspected += 1

        self.diseases.append(disease)

    # ---------------------------------------

    def add_spray(self, amount):

        self.sprayed += 1

        self.chemical_used += amount

    # ---------------------------------------

    def set_distance(self, distance):

        self.distance = distance

    # ---------------------------------------

    def set_remaining_tank(self, percent):

        self.remaining_tank = percent

    # ---------------------------------------

    def finish(self):

        self.end_time = datetime.now()

    # ---------------------------------------

    def save(

        self,

        filename="mission_report.txt"

    ):

        if self.end_time is None:

            self.finish()

        duration = (

            self.end_time -

            self.start_time

        ).total_seconds()

        stats = Counter(self.diseases)

        with open(filename, "w") as file:

            file.write("========== MISSION REPORT ==========\n\n")

            file.write(f"Duration          : {duration:.1f} s\n")

            file.write(f"Distance          : {self.distance:.1f} m\n")

            file.write(f"Plants inspected  : {self.inspected}\n")

            file.write(f"Plants sprayed    : {self.sprayed}\n")

            file.write(f"Chemical used     : {self.chemical_used:.3f} L\n")

            file.write(f"Tank remaining    : {self.remaining_tank:.1f}%\n\n")

            file.write("Disease Statistics\n")

            file.write("------------------\n")

            for disease, count in stats.items():

                file.write(f"{disease:12s}: {count}\n")

        print(f"Mission report saved to {filename}")