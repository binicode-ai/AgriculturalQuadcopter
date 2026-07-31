from agriculture.treatment_database import TreatmentDatabase

db = TreatmentDatabase()

diseases = [

    "Healthy",

    "Rust",

    "Blight",

    "LeafSpot",

    "Mildew",

    "Unknown"

]

print()

print("========== TREATMENT DATABASE ==========")

for disease in diseases:

    info = db.get(disease)

    print()

    print(f"Disease  : {disease}")

    print(f"Spray    : {info['spray']}")

    print(f"Chemical : {info['chemical']}")

    print(f"Duration : {info['duration']} s")

    print(f"Flow     : {info['flow_rate']} L/s")

    print(f"Severity : {info['severity']}")