"""
Dataset manager for crop disease classification.

Author: Biniyam Samuel
"""

from pathlib import Path


class CropDiseaseDataset:

    def __init__(self, root_directory):

        self.root = Path(root_directory)

        self.classes = []

        self.samples = []

    # -------------------------------------

    def scan(self):

        """
        Scan the dataset directory.
        """

        self.classes = []

        self.samples = []

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.root}"
            )

        for class_dir in sorted(self.root.iterdir()):

            if not class_dir.is_dir():
                continue

            self.classes.append(class_dir.name)

            for image_file in class_dir.glob("*"):

                if image_file.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp"
                ]:

                    self.samples.append(

                        (
                            str(image_file),
                            class_dir.name
                        )

                    )

        return self.samples

    # -------------------------------------

    def number_of_classes(self):

        return len(self.classes)

    # -------------------------------------

    def number_of_images(self):

        return len(self.samples)

    # -------------------------------------

    def summary(self):

        print()

        print("========== DATASET ==========")

        print("Classes :", self.number_of_classes())

        print("Images  :", self.number_of_images())

        print()

        for cls in self.classes:

            count = sum(

                label == cls

                for _, label in self.samples

            )

            print(f"{cls:12s}: {count}")

        print("=============================")