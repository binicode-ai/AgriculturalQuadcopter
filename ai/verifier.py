"""
ai/verifier.py

Dataset verification utilities.

Author: Biniyam Samuel
"""

from pathlib import Path
from PIL import Image


class DatasetVerifier:

    def __init__(self, dataset_root):

        self.root = Path(dataset_root)

    # ------------------------------------------------

    def verify(self):

        valid = 0
        invalid = 0

        print("\n========== DATASET VERIFICATION ==========\n")

        for image_path in sorted(

            self.root.rglob("*")

        ):

            if image_path.suffix.lower() not in [

                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff",
                ".webp"

            ]:

                continue

            try:

                with Image.open(image_path) as img:

                    width, height = img.size

                    print(

                        f"OK  {image_path.name:30s}"
                        f"{width}x{height}"

                    )

                valid += 1

            except Exception:

                print(

                    f"BAD {image_path}"

                )

                invalid += 1

        print("\n==========================================")

        print(f"Valid Images   : {valid}")

        print(f"Invalid Images : {invalid}")

        print("==========================================")

        return valid, invalid