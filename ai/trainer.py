"""
ai/trainer.py

PyTorch training pipeline for crop disease classification.

Author: Biniyam Samuel
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader

from ai.cnn import CropDiseaseCNN


class Trainer:

    def __init__(

        self,

        dataset_path,

        batch_size=16,

        learning_rate=0.001,

        epochs=10,

        device=None

    ):

        self.dataset_path = Path(dataset_path)

        self.batch_size = batch_size

        self.learning_rate = learning_rate

        self.epochs = epochs

        self.device = device

        if self.device is None:

            self.device = torch.device(

                "cuda"

                if torch.cuda.is_available()

                else "cpu"

            )

        self.transform = transforms.Compose([

            transforms.Resize((224,224)),

            transforms.RandomHorizontalFlip(),

            transforms.RandomRotation(15),

            transforms.ColorJitter(

                brightness=0.2,

                contrast=0.2,

                saturation=0.2

            ),

            transforms.ToTensor()

        ])

    # ---------------------------------------------

    def build_dataset(self):

        dataset = datasets.ImageFolder(

            self.dataset_path,

            transform=self.transform

        )

        loader = DataLoader(

            dataset,

            batch_size=self.batch_size,

            shuffle=True

        )

        return dataset, loader

    # ---------------------------------------------

    def train(self):

        dataset, loader = self.build_dataset()

        model = CropDiseaseCNN(

            num_classes=len(dataset.classes)

        )

        model.to(self.device)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(

            model.parameters(),

            lr=self.learning_rate

        )

        print()

        print("Training on:", self.device)

        print("Classes:", dataset.classes)

        print()

        for epoch in range(self.epochs):

            running_loss = 0.0

            correct = 0

            total = 0

            model.train()

            for images, labels in loader:

                images = images.to(self.device)

                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(

                    outputs,

                    labels

                )

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(

                    outputs,

                    1

                )

                total += labels.size(0)

                correct += (

                    predicted == labels

                ).sum().item()

            accuracy = 100 * correct / total

            print(

                f"Epoch {epoch+1}/{self.epochs}"

            )

            print(

                f"Loss : {running_loss:.4f}"

            )

            print(

                f"Accuracy : {accuracy:.2f}%"

            )

            print()

        Path(

            "trained_models"

        ).mkdir(

            exist_ok=True

        )

        torch.save(

            model.state_dict(),

            "trained_models/crop_cnn.pth"

        )

        print()

        print("Model saved.")
