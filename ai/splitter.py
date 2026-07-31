"""
ai/splitter.py

Split a labeled image dataset into
train/validation/test folders.

Author: Biniyam Samuel
"""

from pathlib import Path
import shutil
import random


class DatasetSplitter:

    def __init__(
        self,
        source_dir,
        destination_dir,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=42
    ):

        self.source = Path(source_dir)
        self.destination = Path(destination_dir)

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

        random.seed(seed)

    # -----------------------------------------

    def split(self):

        classes = [

            folder.name

            for folder in self.source.iterdir()

            if folder.is_dir()

        ]

        for cls in classes:

            images = []

            for ext in (

                "*.jpg",
                "*.jpeg",
                "*.png",
                "*.bmp",
                "*.tif",
                "*.tiff",
                "*.webp"

            ):

                images.extend(

                    list(

                        (self.source / cls).glob(ext)

                    )

                )

            random.shuffle(images)

            n = len(images)

            train_end = int(

                self.train_ratio * n

            )

            validation_end = train_end + int(

                self.validation_ratio * n

            )

            train = images[:train_end]

            validation = images[train_end:validation_end]

            test = images[validation_end:]

            self.copy(train, "train", cls)

            self.copy(validation, "validation", cls)

            self.copy(test, "test", cls)

            print(

                f"{cls:15s}"

                f"{len(train):5d}"

                f"{len(validation):5d}"

                f"{len(test):5d}"

            )

    # -----------------------------------------

    def copy(

        self,

        image_list,

        split,

        cls

    ):

        folder = (

            self.destination

            / split

            / cls

        )

        folder.mkdir(

            parents=True,

            exist_ok=True

        )

        for image in image_list:

            shutil.copy2(

                image,

                folder / image.name

            )