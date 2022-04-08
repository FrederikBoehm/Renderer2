import json
import numpy as np
import math
from scipy.spatial import ConvexHull

RADIUS = 1000
ACCEL_CLASSES = 10

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

def uniformSampleCircle(radius):
    r = math.sqrt(np.random.uniform()) * radius
    theta = 2 * math.pi * np.random.uniform()
    return np.array([r * math.cos(theta), 0, r * math.sin(theta)])

def getTreeDescription(relevant_models, index, volume_descriptions, volume_max_bounds, camera):
    model = relevant_models[index]
    model_name = model["Name"]
    model_space_height = model["Dimensions"][1]
    expected = model["ExpectedHeight"]
    randomHeight = np.random.normal(expected, expected * 0.01)
    scaling = randomHeight / model_space_height
    bounding_circle_pos = uniformSampleCircle(RADIUS)
    bbmin = np.array(volume_max_bounds[model_name]["BBMin"])
    bbmax = np.array(volume_max_bounds[model_name]["BBMax"])
    # bbmin = np.array(model["BBMin"])
    # bbmax = np.array(model["BBMax"])
    model_pos = (2.0 * bounding_circle_pos - scaling * (bbmin + bbmax)) * 0.5
    model_pos[1] = 0
    model_dimensions = scaling * (bbmax - bbmin)
    bounding_radius = math.sqrt((model_dimensions[0] / 2) ** 2 + (model_dimensions[2] / 2) ** 2)


    lod = selectLOD(camera, model_pos, scaling, volume_descriptions[model["Name"]])
    if lod == "0":
    # if True:
        # dimensions = np.array(model["Dimensions"]) * scaling

        description = {
            "Type": "Mesh",
            "Directory": "../../raytracer/assets/" + model["Path"],
            "Filename": model["Name"] + ".obj",
            "Pos": model_pos,
            "BoundingCirclePos": bounding_circle_pos,
            "Orientation": model["Orientation"],
            "Scaling": [scaling, scaling, scaling],
            "BoundingRadius": bounding_radius
        }
        return description
    else:
        description = {
            "Type": "Medium",
            "Directory": "./filtering/" + model_name + "/",
            "Filename": "filtered_mesh_" + lod + ".nvdb",
            "Pos": model_pos,
            "BoundingCirclePos": bounding_circle_pos,
            "Orientation": model["Orientation"],
            "Scaling": [scaling, scaling, scaling],
            "BoundingRadius": bounding_radius
        }
        return description
        
def vec4(vec, v):
    return np.array([vec[0], vec[1], vec[2], v])

def project(p, camera):
    d = camera["NearPlaneDistance"]
    cam_pos = camera["Pos"]
    # z = cam_pos[2] - p[2]
    z = -p[2]
    x = d / z * p[0]
    y = d / z * p[1]
    return [x, y, -math.copysign(d, z)] # right handed coordinates

def projectLOD(lod, camera, scaling, tree_pos):
    p0 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P0"]) + np.array(tree_pos), 1.0)), camera)
    p1 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P1"]) + np.array(tree_pos), 1.0)), camera)
    p2 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P2"]) + np.array(tree_pos), 1.0)), camera)
    p3 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P3"]) + np.array(tree_pos), 1.0)), camera)
    p4 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P4"]) + np.array(tree_pos), 1.0)), camera)
    p5 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P5"]) + np.array(tree_pos), 1.0)), camera)
    p6 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P6"]) + np.array(tree_pos), 1.0)), camera)
    p7 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P7"]) + np.array(tree_pos), 1.0)), camera)

    return {
        "P0": p0,
        "P1": p1,
        "P2": p2,
        "P3": p3,
        "P4": p4,
        "P5": p5,
        "P6": p6,
        "P7": p7
    }

def outsideFrustum(p, camera):
    pixel_size = camera["PixelSize"]
    return p[2] > 0.0 or math.fabs(p[0]) > camera["SensorWidth"] * 0.5 * pixel_size or math.fabs(p[1]) > camera["SensorHeight"] * 0.5 * pixel_size

def approximateNumVoxels(lod, lod_pos, camera):
    # lod_pos: center of bounding circle
    num_voxels = lod["NumVoxels"]
    return math.ceil(math.pow(num_voxels[0] * num_voxels[1] * num_voxels[2], 2/3))

def selectLOD(camera, tree_pos, scaling, lods):
    # use coarsest volume if outside frustum
    max_lod_size = "0"
    for lod_size in lods:
        if float(lod_size) > float(max_lod_size):
            max_lod_size = lod_size

    max_lod = lods[max_lod_size]
    max_lod_projected = projectLOD(max_lod, camera, scaling, tree_pos)

    points_outside_frustum = 0
    for point_name, point_value in max_lod_projected.items():
        if outsideFrustum(point_value, camera):
            points_outside_frustum += 1

    if points_outside_frustum == 8:
        return max_lod_size

    convex_hull = ConvexHull(np.array([
        [max_lod_projected["P0"][0], max_lod_projected["P0"][1]],
        [max_lod_projected["P1"][0], max_lod_projected["P1"][1]],
        [max_lod_projected["P2"][0], max_lod_projected["P2"][1]],
        [max_lod_projected["P3"][0], max_lod_projected["P3"][1]],
        [max_lod_projected["P4"][0], max_lod_projected["P4"][1]],
        [max_lod_projected["P5"][0], max_lod_projected["P5"][1]],
        [max_lod_projected["P6"][0], max_lod_projected["P6"][1]],
        [max_lod_projected["P7"][0], max_lod_projected["P7"][1]]
    ]))

    pixel_area = camera["PixelSize"] ** 2
    num_covered_pixels = math.ceil(convex_hull.volume / pixel_area) # since ConvexHull is defined in N dimensions, volume gives area in 2d

    best_match = "0"
    for lod_size in sorted(lods):
        if approximateNumVoxels(lods[lod_size], None, camera) > num_covered_pixels:
            best_match = lod_size
        else:
            return best_match

    return best_match


    # world_to_view = camera["WorldToView"]
    #
    #
    # lod = lods["0.400000"]
    # view_dir = (camera["LookAt"] - camera["Pos"]) / np.linalg.norm(camera["LookAt"] - camera["Pos"])
    # view_plane_origin = camera["LookAt"] + view_dir * camera["NearPlaneDistance"]
    # view_plane_origin_cam_space = np.dot(camera["WorldToView"], vec4(view_plane_origin, 1.0))
    # scaling = 0.017187480335105507
    # p0_view = np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P0"]), 1.0))
    # p1_view = np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P1"]), 1.0))
    # p2_view = np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P2"]), 1.0))
    # p3_view = np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P3"]), 1.0))
    # p4_view = np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P4"]), 1.0))
    # p5_view = np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P5"]), 1.0))
    # p6_view = np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P6"]), 1.0))
    # p7_view = np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P7"]), 1.0))
    #
    # p0 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P0"]), 1.0)), camera)
    # p1 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P1"]), 1.0)), camera)
    # p2 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P2"]), 1.0)), camera)
    # p3 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P3"]), 1.0)), camera)
    # p4 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P4"]), 1.0)), camera)
    # p5 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P5"]), 1.0)), camera)
    # p6 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P6"]), 1.0)), camera)
    # p7 = project(np.dot(camera["WorldToView"], vec4(scaling * np.array(lod["P7"]), 1.0)), camera)
    # # world_connection = vec4(lod["P0"], 1.0) - vec4(lod["P7"], 1.0)
    # # view_connection = p0 - p7
    # # world_dist = np.linalg.norm(world_connection)
    # # view_dist = np.linalg.norm(view_connection)
    # # transformed = np.dot(camera["WorldToView"], vec4([0, 0, 1], 1))
    # # transformed2 = np.dot(camera["WorldToView"], vec4([0, 0, 1.00001], 1))
    # return "0.400000"

def checkCollision(object1, object2):
    connection = object1["BoundingCirclePos"] - object2["BoundingCirclePos"]
    distance = np.sqrt(np.dot(connection, connection))

    # return distance < object1["BoundingRadius"] or distance < object2["BoundingRadius"]
    return distance <= object1["BoundingRadius"] + object2["BoundingRadius"]

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
                "Mask": ["FILTER", "RENDER"]
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
    pos = description["BoundingCirclePos"]
    distance = np.sqrt(np.dot(pos, pos))
    bounding_radius = description["BoundingRadius"]
    upper_limit = distance + bounding_radius
    lower_limit = distance - bounding_radius

    class_size = RADIUS / ACCEL_CLASSES
    squared_radius = RADIUS ** 2
    min_class = int(ACCEL_CLASSES * lower_limit ** 2 / squared_radius)
    max_class = int(ACCEL_CLASSES * upper_limit ** 2 / squared_radius)
    return list(range(min_class, min(max_class + 1, ACCEL_CLASSES)))

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
    sensor_width_in_meters = sensor_width * pixel_size
    near_plane_distance = (sensor_width * pixel_size) / (2.0 * math.tan(math.radians(fov) / 2.0))

    pixel_size_x_view = np.dot(worldToView, vec4(pos, 1.0) + pixel_size * u)
    pixel_size_y_view = np.dot(worldToView, vec4(pos, 1.0) + pixel_size * v)
    near_plane_distance_view = np.dot(worldToView, vec4(pos, 1.0) - near_plane_distance * w)

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

def generateRandomForest():
    # tree_cdf = [0.9, 0.98, 1.0] # large, medium, small
    tree_cdf = [0.0, 1.0, 1.0]
    np.random.seed(0)

    with open('relevant_models.json', 'r') as f:
        relevant_models_dict = json.load(f)

    with open('volume_description.json', 'r') as f:
        volume_description_dict = json.load(f)

    volume_max_bounds = getMaxBounds(volume_description_dict)
    camera = getCamera([ -680.0, 25.1, 680.0 ], [ -675.0, 25.0, 675.0 ], [ 0.0, 1.0, 0.0 ], 1920, 1080, 90)

    acceleration = []
    for c in range(ACCEL_CLASSES):
        acceleration.append([])
    for i in range(100000):
    # for i in range(1):

        random_tree_class = np.random.uniform()
        relevant_models = None
        if random_tree_class < tree_cdf[0]:
            relevant_models = relevant_models_dict["Large"]
        elif random_tree_class < tree_cdf[1]:
            relevant_models = relevant_models_dict["Medium"]
        else:
            relevant_models = relevant_models_dict["Small"]

        index = int(np.random.uniform() * len(relevant_models))
        description = getTreeDescription(relevant_models, index, volume_description_dict, volume_max_bounds, camera)
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
                sceneobject["Type"],
                sceneobject["Directory"],
                sceneobject["Filename"],
                tuple(sceneobject["Pos"]),
                tuple(sceneobject["Orientation"]),
                tuple(sceneobject["Scaling"])
                # sceneobject["BoundingRadius"]
            ))

    print(f"Scene has {len(scene)} items")
    toFile(scene, "scenedescription.json")



if __name__ == "__main__":
    generateRandomForest()

