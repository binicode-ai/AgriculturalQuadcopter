"""
agriculture/mission_manager.py

Coordinates AI detection and spraying.

Author: Biniyam Samuel
"""

import time


class AgricultureMissionManager:

    def __init__(

        self,

        detector,

        selector,

        sprayer

    ):

        self.detector = detector
        self.selector = selector
        self.sprayer = sprayer

    # -----------------------------------------

    def process_image(

        self,

        image_path

    ):

        disease, confidence = self.detector.predict(

            image_path

        )

        print()

        print("========== DETECTION ==========")

        print(f"Disease   : {disease}")

        print(f"Confidence: {confidence*100:.2f}%")

        spray = self.selector.should_spray(

            disease,

            confidence

        )

        if spray:

            print()

            print("Decision : SPRAY")

            self.sprayer.spray()

        else:

            print()

            print("Decision : DO NOT SPRAY")

        print("==============================")

        return spray

    # -----------------------------------------

    def process_images(

        self,

        image_list

    ):

        sprayed = 0

        inspected = 0

        for image in image_list:

            inspected += 1

            if self.process_image(image):

                sprayed += 1

            time.sleep(0.2)

        print()

        print("========== SUMMARY ==========")

        print(f"Images inspected : {inspected}")

        print(f"Plants sprayed   : {sprayed}")

        print(f"Tank Remaining   : {self.sprayer.level_percent():.1f}%")

        print("=============================")