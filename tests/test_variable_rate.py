from agriculture.variable_rate import VariableRateController

controller = VariableRateController()

cases = [

    ("Low", 0.90, 2.0, 3.0),

    ("Medium", 0.95, 2.5, 5.0),

    ("High", 0.99, 3.0, 8.0)

]

for severity, confidence, altitude, speed in cases:

    result = controller.compute(

        base_duration=1.5,

        base_flow_rate=0.05,

        severity=severity,

        confidence=confidence,

        altitude=altitude,

        speed=speed

    )

    print("\n==========")

    print(f"Severity   : {severity}")

    print(f"Confidence : {confidence:.2f}")

    print(f"Altitude   : {altitude:.1f} m")

    print(f"Speed      : {speed:.1f} m/s")

    print(f"Duration   : {result['duration']:.2f} s")

    print(f"Flow Rate  : {result['flow_rate']:.3f} L/s")