from agriculture.mission_manager import AgricultureMissionManager
from agriculture.target_selector import TargetSelector
from agriculture.spray_controller import SprayController


class FakeDetector:

    def __init__(self):

        self.index = 0

        self.results = [

            ("Healthy",0.99),

            ("Rust",0.95),

            ("LeafSpot",0.81),

            ("Blight",0.96),

            ("Healthy",0.99),

            ("Mildew",0.93)

        ]

    def predict(self,image):

        result = self.results[self.index]

        self.index += 1

        return result


selector = TargetSelector()

sprayer = SprayController()

detector = FakeDetector()

mission = AgricultureMissionManager(

    detector,

    selector,

    sprayer

)

mission.process_images([

    "img1.jpg",

    "img2.jpg",

    "img3.jpg",

    "img4.jpg",

    "img5.jpg",

    "img6.jpg"

])