import json

with open("../iot_data/trucks.json") as f:
    data = json.load(f)

print(data["TruckA"])