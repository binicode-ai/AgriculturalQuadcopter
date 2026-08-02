from communication.mavlink import MAVLinkInterface

mav = MAVLinkInterface()

mav.wait_heartbeat()

gps = mav.read_gps()

print()

print("GPS")

print(gps)

attitude = mav.read_attitude()

print()

print("ATTITUDE")

print(attitude)