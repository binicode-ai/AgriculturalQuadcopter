from ai.inference import DiseaseInference

classes = [

    "Healthy",
    "Rust",
    "Blight",
    "LeafSpot",
    "Mildew"

]

model = DiseaseInference(

    model_path="trained_models/crop_cnn.pth",

    class_names=classes

)

label, confidence = model.predict(

    "data/sample_leaf.jpg"

)

print()

print("Prediction")

print(label)

print()

print("Confidence")

print(f"{confidence*100:.2f}%")