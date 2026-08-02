from agriculture.mission_report import MissionReport

report = MissionReport()

detections = [

    "Healthy",
    "Rust",
    "Healthy",
    "Blight",
    "Healthy",
    "Rust",
    "LeafSpot",
    "Healthy"

]

for disease in detections:

    report.add_detection(disease)

report.add_spray(0.05)

report.add_spray(0.07)

report.set_distance(245.6)

report.set_remaining_tank(92.5)

report.finish()

report.save()