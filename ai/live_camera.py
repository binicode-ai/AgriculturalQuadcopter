"""
ai/live_camera.py

Live camera disease detection.

Author: Biniyam Samuel
"""

import cv2
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from ai.cnn import CropDiseaseCNN


class LiveDiseaseDetector:

    def __init__(self, model_path, class_names):

        self.class_names = class_names

        self.device = torch.device(

            "cuda"

            if torch.cuda.is_available()

            else "cpu"

        )

        self.model = CropDiseaseCNN(

            num_classes=len(class_names)

        )

        self.model.load_state_dict(

            torch.load(

                model_path,

                map_location=self.device

            )

        )

        self.model.to(self.device)

        self.model.eval()

        self.transform = transforms.Compose([

            transforms.Resize((224,224)),

            transforms.ToTensor()

        ])

    # --------------------------------------------

    def predict_frame(self, frame):

        rgb = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB

        )

        image = Image.fromarray(rgb)

        image = self.transform(

            image

        ).unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():

            output = self.model(image)

            probs = F.softmax(output, dim=1)

            confidence, index = torch.max(

                probs,

                1

            )

        return (

            self.class_names[index.item()],

            confidence.item()

        )

    # --------------------------------------------

    def run(self):

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():

            raise RuntimeError(

                "Cannot open camera."

            )

        while True:

            ret, frame = cap.read()

            if not ret:

                break

            label, conf = self.predict_frame(frame)

            text = (

                f"{label} "

                f"{conf*100:.1f}%"

            )

            cv2.putText(

                frame,

                text,

                (20,40),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0,255,0),

                2

            )

            cv2.imshow(

                "Disease Detection",

                frame

            )

            if cv2.waitKey(1) == 27:

                break

        cap.release()

        cv2.destroyAllWindows()