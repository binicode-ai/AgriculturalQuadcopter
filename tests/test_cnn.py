import torch

from ai.cnn import CropDiseaseCNN

model = CropDiseaseCNN()

print(model)

dummy = torch.randn(

    1,
    3,
    224,
    224

)

output = model(dummy)

print()

print("Output shape")

print(output.shape)