"""
agriculture/spray_logger.py

Logs every spray event to a CSV file.

Author: Biniyam Samuel
"""

import csv
import os
from datetime import datetime


class SprayLogger:

    def __init__(

        self,

        filename="spray_log.csv"

    ):

        self.filename = filename

        if not os.path.exists(self.filename):

            with open(

                self.filename,

                "w",

                newline=""

            ) as file:

                writer = csv.writer(file)

                writer.writerow([

                    "timestamp",

                    "x",

                    "y",

                    "disease",

                    "confidence",

                    "chemical",

                    "duration",

                    "flow_rate"

                ])

    # -----------------------------------------

    def log(

        self,

        x,

        y,

        disease,

        confidence,

        chemical,

        duration,

        flow_rate

    ):

        timestamp = datetime.now().strftime(

            "%Y-%m-%d %H:%M:%S"

        )

        with open(

            self.filename,

            "a",

            newline=""

        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                timestamp,

                x,

                y,

                disease,

                round(confidence, 3),

                chemical,

                round(duration, 2),

                round(flow_rate, 3)

            ])

        print(

            f"Logged spray at ({x:.2f}, {y:.2f})"

        )