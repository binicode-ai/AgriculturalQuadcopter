"""
agriculture/field_mapper.py

Simple field health map visualization.

Author: Biniyam Samuel
"""

import matplotlib.pyplot as plt


class FieldMapper:

    def __init__(self):

        self.points = []

        self.colors = {

            "Healthy": "green",

            "Rust": "orange",

            "Blight": "red",

            "LeafSpot": "yellow",

            "Mildew": "blue",

            "Unknown": "gray"

        }

    # ---------------------------------------------

    def add_observation(

        self,

        x,

        y,

        disease

    ):

        self.points.append(

            (x, y, disease)

        )

    # ---------------------------------------------

    def plot(self):

        plt.figure(figsize=(8, 6))

        for x, y, disease in self.points:

            plt.scatter(

                x,

                y,

                color=self.colors.get(

                    disease,

                    "black"

                ),

                s=80,

                edgecolors="black"

            )

            plt.text(

                x + 0.1,

                y + 0.1,

                disease,

                fontsize=8

            )

        plt.title("Field Health Map")

        plt.xlabel("Field X (m)")

        plt.ylabel("Field Y (m)")

        plt.grid(True)

        plt.axis("equal")

        plt.show()