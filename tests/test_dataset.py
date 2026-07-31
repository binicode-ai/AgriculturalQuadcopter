from ai.dataset import CropDiseaseDataset

dataset = CropDiseaseDataset(
    "datasets/crop_disease/train"
)

dataset.scan()

dataset.summary()