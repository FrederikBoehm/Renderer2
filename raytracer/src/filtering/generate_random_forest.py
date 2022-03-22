import json
import numpy as np
import math

RADIUS = 1000
ACCEL_CLASSES = 10

def uniformSampleCircle(radius):
    r = math.sqrt(np.random.uniform()) * radius
    theta = 2 * math.pi * np.random.uniform()
    return np.array([r * math.cos(theta), 0, r * math.sin(theta)])

def getTreeDescription(relevant_models, index):
    model = relevant_models[index]
    model_space_height = model["Dimensions"][1]
    expected = model["ExpectedHeight"]
    randomHeight = np.random.normal(expected, expected * 0.01)
    scaling = randomHeight / model_space_height
    dimensions = np.array(model["Dimensions"]) * scaling

    description = {
        "Directory": "../../raytracer/assets/" + model["Path"],
        "Filename": model["Name"],
        "Pos": uniformSampleCircle(RADIUS),
        "Orientation": model["Orientation"],
        "Scaling": [scaling, scaling, scaling],
        "BoundingRadius": math.sqrt((dimensions[0] / 2) ** 2 + (dimensions[2] / 2) ** 2)
    }
    return description

def checkCollision(object1, object2):
    connection = object1["Pos"] - object2["Pos"]
    distance = np.sqrt(np.dot(connection, connection))

    return distance < object1["BoundingRadius"] or distance < object2["BoundingRadius"]

def toFile(scene, outPath):
    converted = []
    for sceneobject in scene:
        converted.append({
            "Type": "Mesh",
            "Directory": sceneobject[0],
            "Filename": sceneobject[1],
            "Pos": sceneobject[2],
            "Orientation": sceneobject[3],
            "Scaling": sceneobject[4],
            "Mask": ["FILTER", "RENDER"]
        })

    converted.append({
        "Type": "Circle",
        "Pos": [0.0, 0.0, 0.0],
        "Radius": 3.402823466e+38,
        "Normal": [0.0, 1.0, 0.0],
        "DiffuseReflection": [0.07, 0.07, 0.01],
        "DiffuseRoughness": 0.999,
        "SpecularReflection": [0.9, 0.9, 0.9],
        "AlphaX": 0.99,
        "AlphaY": 0.99,
        "EtaI": 1.00029,
        "EtaT": 1.2
    })

    with open(outPath, 'w') as out:
        json.dump(converted, out, indent=True)

def getRelevantAccelerationClasses(description):
    pos = description["Pos"]
    distance = np.sqrt(np.dot(pos, pos))
    bounding_radius = description["BoundingRadius"]
    upper_limit = distance + bounding_radius
    lower_limit = distance - bounding_radius

    class_size = RADIUS / ACCEL_CLASSES
    squared_radius = RADIUS ** 2
    min_class = int(ACCEL_CLASSES * lower_limit ** 2 / squared_radius)
    max_class = int(ACCEL_CLASSES * upper_limit ** 2 / squared_radius)
    return list(range(min_class, min(max_class + 1, ACCEL_CLASSES)))

def generateRandomForest():
    tree_cdf = [0.9, 0.98, 1.0] # large, medium, small

    with open('relevant_models.json', 'r') as f:
        data = json.load(f)


    acceleration = []
    for c in range(ACCEL_CLASSES):
        acceleration.append([])
    for i in range(100000):

        random_tree_class = np.random.uniform()
        relevant_models = None
        if random_tree_class < tree_cdf[0]:
            relevant_models = data["Large"]
        elif random_tree_class < tree_cdf[1]:
            relevant_models = data["Medium"]
        else:
            relevant_models = data["Small"]

        index = int(np.random.uniform() * len(relevant_models))
        description = getTreeDescription(relevant_models, index)
        acceleration_classes = getRelevantAccelerationClasses(description)

        collide = False
        for acceleration_class in acceleration_classes:
            for sceneobject in acceleration[acceleration_class]:
                if checkCollision(sceneobject, description):
                    collide = True
                    break

        if not collide:
            for acceleration_class in acceleration_classes:
                acceleration[acceleration_class].append(description)

    scene = set()
    for acceleration_class in acceleration:
        for sceneobject in acceleration_class:
            scene.add((
                sceneobject["Directory"],
                sceneobject["Filename"],
                tuple(sceneobject["Pos"]),
                tuple(sceneobject["Orientation"]),
                tuple(sceneobject["Scaling"]),
                sceneobject["BoundingRadius"]
            ))

    print(f"Scene has {len(scene)} items")
    toFile(scene, "scenedescription.json")



if __name__ == "__main__":
    generateRandomForest()

