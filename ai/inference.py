"""
ai/inference.py

Real-time crop disease inference.

Author: Biniyam Samuel
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image

from ai.cnn import CropDiseaseCNN


class DiseaseInference:

    def __init__(

        self,

        model_path,

        class_names,

        device=None

    ):

        if device is None:

            device = torch.device(

                "cuda"

                if torch.cuda.is_available()

                else "cpu"

            )

        self.device = device

        self.class_names = class_names

        self.model = CropDiseaseCNN(

            num_classes=len(class_names)

        )

        self.model.load_state_dict(

            torch.load(

                model_path,

                map_location=device

            )

        )

        self.model.to(device)

        self.model.eval()

        self.transform = transforms.Compose([

            transforms.Resize(

                (224,224)

            ),

            transforms.ToTensor()

        ])

    # ------------------------------------------------

    def predict(

        self,

        image_path

    ):

        image = Image.open(

            image_path

        ).convert("RGB")

        image = self.transform(

            image

        ).unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():

            output = self.model(image)

            probability = F.softmax(

                output,

                dim=1

            )

            confidence, index = torch.max(

                probability,

                1

            )

        return (

            self.class_names[index.item()],

            confidence.item()

        )