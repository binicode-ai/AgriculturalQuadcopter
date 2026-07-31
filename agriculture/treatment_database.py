"""
agriculture/treatment_database.py

Stores treatment recommendations
for crop diseases.

Author: Biniyam Samuel
"""


class TreatmentDatabase:

    def __init__(self):

        self.database = {

            "Healthy": {

                "spray": False,

                "chemical": None,

                "duration": 0.0,

                "flow_rate": 0.0,

                "severity": "None"

            },

            "Rust": {

                "spray": True,

                "chemical": "Fungicide-A",

                "duration": 1.5,

                "flow_rate": 0.05,

                "severity": "Medium"

            },

            "Blight": {

                "spray": True,

                "chemical": "Copper Fungicide",

                "duration": 2.0,

                "flow_rate": 0.07,

                "severity": "High"

            },

            "LeafSpot": {

                "spray": True,

                "chemical": "Broad Spectrum Fungicide",

                "duration": 1.2,

                "flow_rate": 0.05,

                "severity": "Medium"

            },

            "Mildew": {

                "spray": True,

                "chemical": "Sulfur Spray",

                "duration": 1.0,

                "flow_rate": 0.04,

                "severity": "Low"

            }

        }

    # ------------------------------------------

    def get(self, disease):

        return self.database.get(

            disease,

            {

                "spray": False,

                "chemical": None,

                "duration": 0.0,

                "flow_rate": 0.0,

                "severity": "Unknown"

            }

        )