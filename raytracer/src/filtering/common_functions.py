import numpy as np
import math
import json

def getMaxBounds(volume_descriptions):
    max_bounds = {}
    for model_name, values in volume_descriptions.items():
        bbmin = values[list(values.keys())[0]]["BBMin"]
        bbmax = values[list(values.keys())[0]]["BBMax"]
        for voxel_size, lod_data in values.items():
            bbmin = [min(bbmin[0], lod_data["BBMin"][0]), min(bbmin[1], lod_data["BBMin"][1]), min(bbmin[2], lod_data["BBMin"][2])]
            bbmax = [max(bbmax[0], lod_data["BBMax"][0]), max(bbmax[1], lod_data["BBMax"][1]), max(bbmax[2], lod_data["BBMax"][2])]

        max_bounds[model_name] = {}
        max_bounds[model_name]["BBMin"] = bbmin
        max_bounds[model_name]["BBMax"] = bbmax

    return max_bounds

def lookAt(eye, center, up):
    eye = np.array(eye)
    center = np.array(center)
    up = np.array(up)

    g = center - eye
    w = - g / np.linalg.norm(g)
    u = np.cross(up, w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)

    M1 = np.array([[u[0], u[1], u[2], 0],
                   [v[0], v[1], v[2], 0],
                   [w[0], w[1], w[2], 0],
                   [0, 0, 0, 1]]) # numpy is row major
    M2 = np.array([[1, 0, 0, -eye[0]],
                   [0, 1, 0, -eye[1]],
                   [0, 0, 1, -eye[2]],
                   [0, 0, 0, 1]])
    return np.matmul(M1, M2)

def getCamera(pos, lookat, up, sensor_width, sensor_height, fov):
    worldToView = lookAt(pos, lookat, up)
    viewToWorld = np.linalg.inv(worldToView)

    u = np.dot(viewToWorld, [1, 0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.dot(viewToWorld, [0, 1, 0, 0])
    v /= np.linalg.norm(v)
    w = np.dot(viewToWorld, [0, 0, 1, 0])
    w /= np.linalg.norm(w)

    pixel_size = 1.e-5
    near_plane_distance = (sensor_width * pixel_size) / (2.0 * math.tan(math.radians(fov) / 2.0))

    camera = {
        "Pos": np.array(pos),
        "LookAt": np.array(lookat),
        "Up": np.array(up),
        "WorldToView": worldToView,
        "ViewToWorld": viewToWorld,
        "SensorWidth": sensor_width,
        "SensorHeight": sensor_height,
        "Fov": fov,
        "NearPlaneDistance": near_plane_distance,
        "PixelSize": pixel_size
    }
    return camera

def toFile(scene, outPath):
    converted = []
    for sceneobject in scene:
        if sceneobject[0] == "Mesh":
            converted.append({
                "Type": "Mesh",
                "Directory": sceneobject[1],
                "Filename": sceneobject[2],
                "Pos": sceneobject[3],
                "Orientation": sceneobject[4],
                "Scaling": sceneobject[5],
                "Mask": ["RENDER"]
            })
        else:
            converted.append({
                "Type": "Medium",
                "Path": sceneobject[1] + "/" + sceneobject[2],
                "SigmaA": [0.0, 0.0, 0.0],
                "SigmaS": [10.0, 10.0, 10.0],
                "Pos": sceneobject[3],
                "Orientation": sceneobject[4],
                "Scaling": sceneobject[5],
                "Mask": ["RENDER"]
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