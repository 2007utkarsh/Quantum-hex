
from services.database import init_db, insert_telemetry
# Lucknow district prototype telemetry points. Replace with a real AVL/GPS feed in deployment.
buses=[
("BUS-104",26.8467,80.9462,32,"Shaheed Path"),
("BUS-117",26.8590,80.9490,27,"Faizabad Road"),
("BUS-121",26.8720,80.9340,22,"Gomti Nagar Extension"),
("BUS-133",26.8320,80.9180,35,"Hazratganj"),
("BUS-141",26.8750,80.9810,18,"Kanpur Road"),
]
init_db()
for b in buses:
    insert_telemetry({"bus_id":b[0],"lat":b[1],"lng":b[2],"speed_kmh":b[3],"route":b[4],"status":"ACTIVE"})
print("Seeded prototype fleet telemetry.")
