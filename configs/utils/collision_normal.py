import numpy as np

def compute_distance(obj1, obj2):
    c1 = np.array([(obj1['xmin'] + obj1['xmax']) / 2,
                   (obj1['ymin'] + obj1['ymax']) / 2])
    c2 = np.array([(obj2['xmin'] + obj2['xmax']) / 2,
                   (obj2['ymin'] + obj2['ymax']) / 2])
    return np.linalg.norm(c1 - c2)


def detect_collisions(objects, threshold=100):
    vehicles = [o for o in objects if o['name'] in ['car', 'bus', 'bike']]
    collisions = []

    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            if compute_distance(vehicles[i], vehicles[j]) < threshold:
                collisions.append((vehicles[i], vehicles[j]))

    return collisions
