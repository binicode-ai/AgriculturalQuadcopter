"""
ai/downloader.py

Dataset preparation utilities.

Author: Biniyam Samuel
"""

from pathlib import Path


class DatasetDownloader:

    def __init__(self):

        self.root = Path("datasets/crop_disease")

    # -----------------------------------------

    def create_structure(self):

        classes = [

            "Healthy",
            "Rust",
            "Blight",
            "LeafSpot",
            "Mildew"

        ]

        for split in [

            "train",
            "validation",
            "test"

        ]:

            for cls in classes:

                folder = self.root / split / cls

                folder.mkdir(
                    parents=True,
                    exist_ok=True
                )

        print("Dataset folders verified.")

    # -----------------------------------------

    def statistics(self):

        print()

        print("========== DATASET ==========")

        total = 0

        for split in [

            "train",
            "validation",
            "test"

        ]:

            print()

            print(split.upper())

            split_total = 0

            for folder in sorted(

                (self.root / split).iterdir()

            ):

                if not folder.is_dir():
                    continue

                count = len(

                    list(folder.glob("*"))

                )

                split_total += count

                print(

                    f"{folder.name:12s}: {count}"

                )

            print(

                f"Total: {split_total}"

            )

            total += split_total

        print()

        print(f"Grand Total: {total}")

        print("=============================")