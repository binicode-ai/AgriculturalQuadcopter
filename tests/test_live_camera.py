from ai.live_camera import LiveDiseaseDetector

classes = [

    "Healthy",

    "Rust",

    "Blight",

    "LeafSpot",

    "Mildew"

]

detector = LiveDiseaseDetector(

    "trained_models/crop_cnn.pth",

    classes

)

detector.run()